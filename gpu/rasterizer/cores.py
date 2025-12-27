from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out

from gpu.utils.stream import WideStreamOutput

from ..utils import math as gpu_math
from ..utils.fixed import Value
from ..utils.layouts import RasterizerLayout, num_textures
from ..utils.types import FixedPoint, PrimitiveType


class PrimitiveClipper(wiring.Component):
    """Primitive clipper for rasterizer stage.

    - Input: stream of `RasterizerLayout` vertices (assembled order depends on primitive type).
    - Output: stream of `RasterizerLayout` vertices with primitives fully inside the clip volume.
    - Registers: primitive type (point/line/triangle), cull face, winding order.
    - Culling: applied for triangles only (front/back based on area sign and winding).
        - Clipping: trivial accept/reject against NDC cube (after perspective divide):
            -1 <= x_ndc,y_ndc,z_ndc <= 1. No polygon splitting.
    """

    is_vertex: In(stream.Signature(RasterizerLayout))
    os_vertex: Out(stream.Signature(RasterizerLayout))

    prim_type: In(PrimitiveType)
    ready: Out(1)

    def elaborate(self, platform):
        m = Module()
        # Reciprocal unit for t computation (t = num / den = num * inv(den))
        m.submodules.inv = inv = gpu_math.FixedPointInv(FixedPoint, steps=4)

        buf = Array(Signal(RasterizerLayout) for _ in range(3))
        idx = Signal(range(3))
        needed = Signal(range(4))

        # Clipping work buffers (up to 9 vertices after clipping against 6 planes)
        clip_buf = Array(
            Array(Signal(RasterizerLayout) for _ in range(9)) for _ in range(2)
        )
        clip_count = Array(Signal(range(10)) for _ in range(2))
        clip_src = Signal()  # ping-pong buffer index
        clip_plane = Signal(range(6))  # current plane being clipped against
        clip_idx = Signal(range(9))  # current vertex index during clipping
        # Interpolation helper signals
        t_num = Signal(FixedPoint)
        t_den = Signal(FixedPoint)
        t_recip = Signal(FixedPoint)
        emit_next = Signal()  # entering case: emit intersection and next vertex
        edge_curr_idx = Signal(range(9))
        edge_next_idx = Signal(range(9))

        # Primitive vertex count based on register
        with m.Switch(self.prim_type):
            with m.Case(PrimitiveType.POINTS):
                m.d.comb += needed.eq(1)
            with m.Case(PrimitiveType.LINES):
                m.d.comb += needed.eq(2)
            with m.Default():
                m.d.comb += needed.eq(3)

        m.submodules.w_out = w_out = WideStreamOutput(self.os_vertex.p.shape(), 3)
        wiring.connect(m, wiring.flipped(self.os_vertex), w_out.o)

        with m.If(w_out.i.ready):
            m.d.sync += w_out.i.valid.eq(0)
        with m.If(w_out.n.ready):
            m.d.sync += w_out.n.valid.eq(0)

        with m.FSM():
            with m.State("COLLECT"):
                m.d.comb += [self.is_vertex.ready.eq(1), self.ready.eq(1)]
                with m.If(self.is_vertex.valid):
                    m.d.sync += buf[idx].eq(self.is_vertex.payload)
                    with m.If(idx == (needed - 1)):
                        m.d.sync += idx.eq(0)
                        m.next = "CHECK"
                    with m.Else():
                        m.d.sync += idx.eq(idx + 1)

            with m.State("CHECK"):
                # Compute clip codes for trivial accept/reject (no polygon splitting).
                # Helper function to compute clip code for a vertex
                def compute_clip_code(vtx):
                    x, y, z, w = vtx.position_ndc
                    bits = [
                        x > w,  # +x
                        x < -w,  # -x
                        y > w,  # +y
                        y < -w,  # -y
                        z > w,  # +z
                        z < -w,  # -z
                    ]
                    return Cat(bits)

                codes = Array(Signal(6) for _ in range(3))
                for i in range(3):
                    m.d.comb += codes[i].eq(compute_clip_code(buf[i]))

                m.d.sync += [
                    Print(
                        Format(
                            "vtx0: {}, vtx1: {}, vtx2: {}",
                            buf[0].position_ndc,
                            buf[1].position_ndc,
                            buf[2].position_ndc,
                        )
                    ),
                    Print(
                        Format(
                            "Clip codes: {:06b}, {:06b}, {:06b}",
                            codes[0],
                            codes[1],
                            codes[2],
                        )
                    ),
                ]
                with m.If((codes[0] & codes[1] & codes[2]) != 0):
                    m.d.sync += Print("Trivial reject")
                    # Fully outside; drop primitive.
                    m.next = "COLLECT"
                with m.Elif((codes[0] | codes[1] | codes[2]) == 0):
                    # Fully inside; forward primitive.
                    with m.If(~w_out.i.valid & ~w_out.n.valid):
                        m.d.sync += Print("Trivial accept")
                        m.d.sync += [
                            w_out.i.p[0].eq(buf[0]),
                            w_out.i.p[1].eq(buf[1]),
                            w_out.i.p[2].eq(buf[2]),
                            w_out.n.p.eq(needed),
                            w_out.i.valid.eq(1),
                            w_out.n.valid.eq(1),
                        ]
                        m.next = "COLLECT"
                with m.Else():
                    # Needs clipping (only triangles and lines should reach here).
                    m.next = "CLIP"

            with m.State("CLIP"):
                # Sutherland-Hodgman clipping for triangles only
                # Initialize first buffer with triangle vertices
                with m.If(needed == 3):
                    m.d.sync += [
                        clip_buf[0][0].eq(buf[0]),
                        clip_buf[0][1].eq(buf[1]),
                        clip_buf[0][2].eq(buf[2]),
                        clip_count[0].eq(3),
                        clip_plane.eq(0),
                        clip_src.eq(0),
                    ]
                    m.next = "CLIP_PLANE"
                with m.Else():
                    # TODO: implement line clipping (Cohen-Sutherland or Liang-Barsky)
                    m.next = "COLLECT"

            with m.State("CLIP_PLANE"):
                # Clip against current plane
                # Planes: 0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z
                src = clip_src
                dst = ~clip_src
                m.d.sync += Print(
                    Format(
                        "CLIP_PLANE enter plane {} src {} dst {} count {}",
                        clip_plane,
                        src,
                        dst,
                        clip_count[src],
                    )
                )

                # If source polygon is empty or clipped away, skip to emit
                with m.If(clip_count[src] == 0):
                    m.next = "CLIP_EMIT"
                with m.Elif(clip_plane > 5):
                    # All planes processed
                    m.next = "CLIP_EMIT"
                with m.Else():
                    m.d.sync += [
                        clip_count[dst].eq(0),
                        clip_idx.eq(0),
                    ]
                    m.next = "CLIP_EDGE"

            with m.State("CLIP_EDGE"):
                # Clip edge across current plane
                src = clip_src
                dst = ~clip_src
                curr_idx = clip_idx
                next_idx = Mux(clip_idx == clip_count[src] - 1, 0, clip_idx + 1)

                curr_v = clip_buf[src][curr_idx]
                next_v = clip_buf[src][next_idx]

                m.d.sync += Print(
                    Format(
                        "CLIP_EDGE plane {} src {} curr {} next {} count {}",
                        clip_plane,
                        src,
                        curr_idx,
                        next_idx,
                        clip_count[src],
                    )
                )

                c_x, c_y, c_z, c_w = curr_v.position_ndc
                n_x, n_y, n_z, n_w = next_v.position_ndc

                m.d.sync += Print(
                    Format(
                        "   curr ({}, {}, {}, {}), next ({}, {}, {}, {})",
                        c_x,
                        c_y,
                        c_z,
                        c_w,
                        n_x,
                        n_y,
                        n_z,
                        n_w,
                    )
                )

                # Compute distance from plane in clip space using ±w comparisons
                # plane 0: +x (x <= w) => dist = w - x
                # plane 1: -x (x >= -w) => dist = x + w
                # plane 2: +y (y <= w) => dist = w - y
                # plane 3: -y (y >= -w) => dist = y + w
                # plane 4: +z (z <= w) => dist = w - z (far)
                # plane 5: -z (z >= -w) => dist = z + w (near)
                curr_dist = Signal(FixedPoint)
                next_dist = Signal(FixedPoint)

                with m.Switch(clip_plane):
                    with m.Case(0):
                        m.d.comb += [
                            curr_dist.eq(c_w - c_x),
                            next_dist.eq(n_w - n_x),
                        ]
                    with m.Case(1):
                        m.d.comb += [
                            curr_dist.eq(c_x + c_w),
                            next_dist.eq(n_x + n_w),
                        ]
                    with m.Case(2):
                        m.d.comb += [
                            curr_dist.eq(c_w - c_y),
                            next_dist.eq(n_w - n_y),
                        ]
                    with m.Case(3):
                        m.d.comb += [
                            curr_dist.eq(c_y + c_w),
                            next_dist.eq(n_y + n_w),
                        ]
                    with m.Case(4):
                        m.d.comb += [
                            curr_dist.eq(c_w - c_z),
                            next_dist.eq(n_w - n_z),
                        ]
                    with m.Case(5):
                        m.d.comb += [
                            curr_dist.eq(c_z + c_w),
                            next_dist.eq(n_z + n_w),
                        ]

                curr_inside = curr_dist >= 0
                next_inside = next_dist >= 0

                m.d.sync += Print(
                    Format(
                        "   dists {} {} inside {} {}",
                        curr_dist,
                        next_dist,
                        curr_inside,
                        next_inside,
                    )
                )

                out_count = clip_count[dst]

                # Prepare indices for use across states
                m.d.sync += [
                    edge_curr_idx.eq(curr_idx),
                    edge_next_idx.eq(next_idx),
                ]

                # Both inside: emit next vertex and move to next edge
                with m.If(curr_inside & next_inside):
                    m.d.sync += [
                        clip_buf[dst][out_count].eq(next_v),
                        clip_count[dst].eq(out_count + 1),
                    ]
                    # Move to next edge
                    with m.If(clip_idx == clip_count[src] - 1):
                        # Done with this plane
                        m.d.sync += clip_src.eq(dst)
                        with m.If(clip_plane == 5):
                            # All planes done
                            m.next = "CLIP_EMIT"
                        with m.Else():
                            m.d.sync += clip_plane.eq(clip_plane + 1)
                            m.next = "CLIP_PLANE"
                    with m.Else():
                        m.d.sync += clip_idx.eq(clip_idx + 1)
                # Exiting: emit intersection
                with m.Elif(curr_inside & ~next_inside):
                    # Compute t via reciprocal: request inv(t_den)
                    m.d.sync += [
                        t_num.eq(curr_dist),
                        t_den.eq(curr_dist - next_dist),
                        emit_next.eq(0),
                    ]
                    m.next = "CLIP_INV_REQ"
                # Entering: emit intersection and next vertex
                with m.Elif(~curr_inside & next_inside):
                    # Compute t via reciprocal: request inv(t_den)
                    m.d.sync += [
                        t_num.eq(curr_dist),
                        t_den.eq(curr_dist - next_dist),
                        emit_next.eq(1),
                    ]
                    m.next = "CLIP_INV_REQ"
                # Both outside: emit nothing, move to next edge
                with m.Else():
                    # Move to next edge
                    with m.If(clip_idx == clip_count[src] - 1):
                        # Done with this plane
                        m.d.sync += clip_src.eq(dst)
                        with m.If(clip_plane == 5):
                            # All planes done
                            m.next = "CLIP_EMIT"
                        with m.Else():
                            m.d.sync += clip_plane.eq(clip_plane + 1)
                            m.next = "CLIP_PLANE"
                    with m.Else():
                        m.d.sync += clip_idx.eq(clip_idx + 1)

            # Request reciprocal for t_den
            with m.State("CLIP_INV_REQ"):
                m.d.comb += [
                    inv.i.valid.eq(1),
                    inv.i.payload.eq(t_den),
                ]
                with m.If(inv.i.ready):
                    m.next = "CLIP_INV_WAIT"

            with m.State("CLIP_INV_WAIT"):
                m.d.comb += inv.o.ready.eq(1)
                with m.If(inv.o.valid):
                    m.d.sync += t_recip.eq(inv.o.p)
                    m.next = "CLIP_APPLY"

            with m.State("CLIP_APPLY"):
                # Apply intersection using t = t_num * t_recip
                src = clip_src
                dst = ~clip_src
                curr_idx = edge_curr_idx
                next_idx = edge_next_idx
                curr_v = clip_buf[src][curr_idx]
                next_v = clip_buf[src][next_idx]
                out_count = clip_count[dst]

                m.d.sync += Print(
                    Format(
                        "CLIP_APPLY plane {} src {} dst {} curr {} next {} out {} emit_next {}",
                        clip_plane,
                        src,
                        dst,
                        curr_idx,
                        next_idx,
                        out_count,
                        emit_next,
                    )
                )

                t = Signal(FixedPoint)
                t_full = t_num * t_recip
                # Constrain t back to the base fixed-point width to avoid overflow
                m.d.comb += t.eq(t_full.reshape(FixedPoint.f_bits))

                def lerp(a, b, t):
                    # Keep intermediate products at the base fractional precision
                    a_v = Value.cast(a, FixedPoint.f_bits)
                    b_v = Value.cast(b, FixedPoint.f_bits)
                    diff = (b_v - a_v).reshape(FixedPoint.f_bits)
                    prod = (t * diff).reshape(FixedPoint.f_bits)
                    return (a_v + prod).reshape(FixedPoint.f_bits)

                def lerp_a(a, b, t, n):
                    return [lerp(a[i], b[i], t) for i in range(n)]

                pos = lerp_a(curr_v.position_ndc, next_v.position_ndc, t, 4)
                col = lerp_a(curr_v.color, next_v.color, t, 4)

                # Force the intersected coordinate to lie exactly on the clipping plane
                # to avoid tiny fixed-point overshoot (e.g. 1.00012 instead of 1.0).
                pos_clamped = [Signal.like(pos[i]) for i in range(len(pos))]
                m.d.comb += [
                    pos_clamped[0].eq(
                        Mux(
                            clip_plane == 0,
                            pos[3],
                            Mux(clip_plane == 1, -pos[3], pos[0]),
                        )
                    ),
                    pos_clamped[1].eq(
                        Mux(
                            clip_plane == 2,
                            pos[3],
                            Mux(clip_plane == 3, -pos[3], pos[1]),
                        )
                    ),
                    pos_clamped[2].eq(
                        Mux(
                            clip_plane == 4,
                            pos[3],
                            Mux(clip_plane == 5, -pos[3], pos[2]),
                        )
                    ),
                    pos_clamped[3].eq(pos[3]),
                ]

                # Build interpolated texcoords
                m.d.sync += Print(Format("      t {}", t))

                # Assign individual fields directly
                for i in range(4):
                    m.d.sync += (
                        clip_buf[dst][out_count].position_ndc[i].eq(pos_clamped[i])
                    )
                for i in range(4):
                    m.d.sync += clip_buf[dst][out_count].color[i].eq(col[i])
                for ti in range(num_textures):
                    tex = lerp_a(curr_v.texcoords[ti], next_v.texcoords[ti], t, 4)
                    for i in range(4):
                        m.d.sync += clip_buf[dst][out_count].texcoords[ti][i].eq(tex[i])
                m.d.sync += clip_buf[dst][out_count].front_facing.eq(
                    curr_v.front_facing
                )

                with m.If(emit_next):
                    m.d.sync += [
                        clip_buf[dst][out_count + 1].eq(next_v),
                        clip_count[dst].eq(out_count + 2),
                    ]
                with m.Else():
                    m.d.sync += [
                        clip_count[dst].eq(out_count + 1),
                    ]

                # Continue to next edge or next plane
                with m.If(clip_idx == clip_count[src] - 1):
                    m.d.sync += clip_src.eq(dst)
                    with m.If(clip_plane == 5):
                        m.next = "CLIP_EMIT"
                    with m.Else():
                        m.d.sync += clip_plane.eq(clip_plane + 1)
                        m.next = "CLIP_PLANE"
                with m.Else():
                    m.d.sync += clip_idx.eq(clip_idx + 1)
                    m.next = "CLIP_EDGE"

            with m.State("CLIP_EMIT"):
                # Emit clipped polygon as triangle fan
                final_buf = clip_src
                final_count = clip_count[final_buf]

                m.d.sync += Print(
                    Format(
                        "CLIP_EMIT count {} plane {} src {}",
                        final_count,
                        clip_plane,
                        final_buf,
                    )
                )

                with m.If(final_count < 3):
                    # Clipped to nothing
                    m.next = "COLLECT"
                with m.Else():
                    # Emit first triangle
                    m.d.sync += clip_idx.eq(2)
                    m.next = "CLIP_OUTPUT"

            with m.State("CLIP_OUTPUT"):
                # Output triangle as fan: (0, idx-1, idx)
                final_buf = clip_src
                final_count = clip_count[final_buf]

                with m.If(~w_out.i.valid & ~w_out.n.valid):
                    out_vertices = Signal.like(w_out.i.p)
                    m.d.comb += [
                        out_vertices[0].eq(clip_buf[final_buf][0]),
                        out_vertices[1].eq(clip_buf[final_buf][clip_idx - 1]),
                        out_vertices[2].eq(clip_buf[final_buf][clip_idx]),
                    ]
                    m.d.sync += [
                        w_out.i.p.eq(out_vertices),
                        w_out.i.valid.eq(1),
                        w_out.n.p.eq(3),
                        w_out.n.valid.eq(1),
                    ]

                    # Check if this is the last triangle
                    with m.If(clip_idx == final_count - 1):
                        # Done, go back to COLLECT
                        m.next = "COLLECT"
                    with m.Else():
                        # More triangles to emit, increment and stay in CLIP_OUTPUT
                        m.d.sync += clip_idx.eq(clip_idx + 1)

        return m
