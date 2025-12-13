import pytest
from amaranth.sim import Simulator

from gpu.primitive_assembly.cores import PrimitiveAssembly
from gpu.utils import fixed
from gpu.utils.types import CullFace, FixedPoint, FrontFace, PrimitiveType

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench


def make_pa_vertex(pos, color, color_back=None):
    return {
        "position_proj": pos,
        "texcoords": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "color": color,
        "color_back": color_back if color_back is not None else color,
    }


def assert_rasterizer_vertex(payload, pos, color, front):
    got_pos = [c.as_float() for c in payload.position_ndc]
    got_color = [c.as_float() for c in payload.color]
    assert got_pos == pytest.approx(pos)
    assert got_color == pytest.approx(color)
    assert int(payload.front_facing) == front


@pytest.mark.parametrize(
    "pos,color",
    [
        ([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
        ([0.5, -0.5, 0.5, 1.0], [0.0, 1.0, 0.0, 1.0]),
        ([-1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
    ],
)
def test_points_passthrough_hypothesis(pos, color):
    # Force w to 1.0 to match NDC expectations
    pos = [fixed.Const(v, FixedPoint).as_float() for v in pos]
    color = [fixed.Const(v, FixedPoint).as_float() for v in color]

    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    t.set_csrs(
        dut.csr_bus,
        [
            ((("prim_type",),), PrimitiveType.POINTS),
            ((("prim_cull",),), CullFace.NONE),
            ((("prim_winding",),), FrontFace.CCW),
        ],
        "dut",
    )

    vertices = [make_pa_vertex(pos, color)]

    async def checker(ctx, results):
        assert len(results) == 1
        assert_rasterizer_vertex(results[0], pos, color, 1)

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=t.initialize_csrs,
        input_stream=dut.is_vertex,
        input_data=vertices,
        output_stream=dut.os_primitive,
        output_data_checker=checker,
        idle_for=50,
    )

    sim.run()


def test_lines_passthrough():
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    t.set_csrs(
        dut.csr_bus,
        [
            ((("prim_type",),), PrimitiveType.LINES),
            ((("prim_cull",),), CullFace.NONE),
            ((("prim_winding",),), FrontFace.CCW),
        ],
        "dut",
    )

    line = [
        make_pa_vertex([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
        make_pa_vertex([0.5, -0.5, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]),
    ]

    async def checker(ctx, results):
        assert len(results) == 2
        assert_rasterizer_vertex(
            results[0], line[0]["position_proj"], line[0]["color"], 1
        )
        assert_rasterizer_vertex(
            results[1], line[1]["position_proj"], line[1]["color"], 1
        )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=t.initialize_csrs,
        input_stream=dut.is_vertex,
        input_data=line,
        output_stream=dut.os_primitive,
        output_data_checker=checker,
        idle_for=50,
    )

    sim.run()


@pytest.mark.parametrize("front_face", [FrontFace.CCW, FrontFace.CW])
@pytest.mark.parametrize(
    "tri, winding_order, expected_ff, expected_colors",
    [
        # CCW positions -> CCW winding; expected_ff depends on FrontFace setting
        (
            [
                make_pa_vertex(
                    [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
                ),
                make_pa_vertex(
                    [1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
                ),
                make_pa_vertex(
                    [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
                ),
            ],
            "CCW",
            {FrontFace.CCW: 1, FrontFace.CW: 0},
            {
                FrontFace.CCW: ["color", "color", "color"],
                FrontFace.CW: ["color_back", "color_back", "color_back"],
            },
        ),
        # CW positions -> CW winding; expected_ff depends on FrontFace setting
        (
            [
                make_pa_vertex(
                    [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
                ),
                make_pa_vertex(
                    [0.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
                ),
                make_pa_vertex(
                    [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
                ),
            ],
            "CW",
            {FrontFace.CCW: 0, FrontFace.CW: 1},
            {
                FrontFace.CCW: ["color_back", "color_back", "color_back"],
                FrontFace.CW: ["color", "color", "color"],
            },
        ),
    ],
)
def test_triangles_winding_and_front_face(
    tri, winding_order, expected_ff, expected_colors, front_face
):
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    t.set_csrs(
        dut.csr_bus,
        [
            ((("prim_type",),), PrimitiveType.TRIANGLES),
            ((("prim_cull",),), CullFace.NONE),
            ((("prim_winding",),), front_face),
        ],
        "dut",
    )

    async def checker(ctx, results):
        assert len(results) == 3
        ff_val = expected_ff[front_face]
        cols = expected_colors[front_face]
        for i in range(3):
            use_key = cols[i]
            exp_color = tri[i][use_key]
            assert_rasterizer_vertex(
                results[i], tri[i]["position_proj"], exp_color, ff_val
            )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=t.initialize_csrs,
        input_stream=dut.is_vertex,
        input_data=tri,
        output_stream=dut.os_primitive,
        output_data_checker=checker,
        idle_for=50,
    )

    sim.run()
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    t.set_csrs(
        dut.csr_bus,
        [
            ((("prim_type",),), PrimitiveType.TRIANGLES),
            ((("prim_cull",),), CullFace.NONE),
            ((("prim_winding",),), FrontFace.CCW),
        ],
        "dut",
    )

    # CCW winding => front-facing with default FrontFace.CCW
    tri = [
        make_pa_vertex(
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
        ),
        make_pa_vertex(
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
        ),
        make_pa_vertex(
            [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
        ),
    ]

    async def checker(ctx, results):
        assert len(results) == 3
        assert_rasterizer_vertex(
            results[0], tri[0]["position_proj"], tri[0]["color"], 1
        )
        assert_rasterizer_vertex(
            results[1], tri[1]["position_proj"], tri[1]["color"], 1
        )
        assert_rasterizer_vertex(
            results[2], tri[2]["position_proj"], tri[2]["color"], 1
        )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=t.initialize_csrs,
        input_stream=dut.is_vertex,
        input_data=tri,
        output_stream=dut.os_primitive,
        output_data_checker=checker,
        idle_for=30,
    )

    sim.run()


def test_triangle_back_face_culled():
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    # CW winding with CCW front-face definition => back-facing
    tri = [
        make_pa_vertex(
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
        ),
        make_pa_vertex(
            [0.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
        ),
        make_pa_vertex(
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
        ),
    ]

    t.set_csrs(
        dut.csr_bus,
        [
            ((("prim_type",),), PrimitiveType.TRIANGLES),
            ((("prim_cull",),), CullFace.BACK),
            ((("prim_winding",),), FrontFace.CCW),
        ],
        "dut",
    )

    async def checker(ctx, results):
        # Back-face culled => no output
        assert results == []

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=t.initialize_csrs,
        input_stream=dut.is_vertex,
        input_data=tri,
        output_stream=dut.os_primitive,
        output_data_checker=checker,
        idle_for=30,
    )

    sim.run()


def test_triangle_back_face_uses_back_color():
    dut = PrimitiveAssembly()
    t = SimpleTestbench(dut)

    t.set_csrs(
        dut.csr_bus,
        [
            ((("prim_type",),), PrimitiveType.TRIANGLES),
            ((("prim_cull",),), CullFace.NONE),
            ((("prim_winding",),), FrontFace.CCW),
        ],
        "dut",
    )

    tri = [
        make_pa_vertex(
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]
        ),
        make_pa_vertex(
            [0.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0]
        ),
        make_pa_vertex(
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 1.0]
        ),
    ]

    async def checker(ctx, results):
        assert len(results) == 3
        assert_rasterizer_vertex(
            results[0], tri[0]["position_proj"], tri[0]["color_back"], 0
        )
        assert_rasterizer_vertex(
            results[1], tri[1]["position_proj"], tri[1]["color_back"], 0
        )
        assert_rasterizer_vertex(
            results[2], tri[2]["position_proj"], tri[2]["color_back"], 0
        )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        init_process=t.initialize_csrs,
        input_stream=dut.is_vertex,
        input_data=tri,
        output_stream=dut.os_primitive,
        output_data_checker=checker,
        idle_for=30,
    )

    sim.run()
