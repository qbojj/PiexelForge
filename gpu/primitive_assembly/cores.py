from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out
from amaranth_soc import csr

from ..utils.layouts import PrimitiveAssemblyLayout, RasterizerLayout
from ..utils.types import PrimitiveType


class PrimitiveAssembly(wiring.Component):
    """Primitive assembly core

    Assembles incoming shaded vertices into primitives for rasterization stage.

    Input: ShadingVertexLayout
    Output: PrimitiveAssemblyLayout
    """

    is_vertex: In(stream.Signature(PrimitiveAssemblyLayout))
    os_primitive: Out(stream.Signature(RasterizerLayout))

    ready: Out(1)

    class PrimitiveReg(csr.Register, access="rw"):
        type: csr.Field(csr.action.RW, PrimitiveType)

    def __init__(self):
        super().__init__()
        regs = csr.Builder(addr_width=8, data_width=8)
        self.prim_type = regs.add(
            name="prim_type", reg=self.PrimitiveReg(), offset=0x00
        )
        self.csr_bridge = csr.Bridge(regs.as_memory_map())
        self.csr_bus = self.csr_bridge.bus

    def elaborate(self, platform):
        m = Module()

        m.d.comb += self.ready.eq(~self.os_primitive.valid)

        with m.If(self.os_primitive.ready):
            m.d.sync += self.os_primitive.valid.eq(0)

        # TODO: Implement assembly for different primitive types

        with m.If(~self.is_vertex.valid):
            m.d.comb += self.is_vertex.ready.eq(1)

            m.d.sync += [
                self.os_primitive.valid.eq(1),
                self.os_primitive.p.position_proj.eq(self.is_vertex.p.position_proj),
                self.os_primitive.p.texcoords.eq(self.is_vertex.p.texcoords),
                self.os_primitive.p.color.eq(self.is_vertex.p.color),
                self.os_primitive.p.front_facing.eq(1),
            ]

            with m.Switch(self.prim_type.f.type.data):
                with m.Case(PrimitiveType.POINTS, PrimitiveType.LINES):
                    m.d.sync += [
                        self.os_primitive.p.color.eq(self.is_vertex.p.color),
                    ]
                with m.Case(PrimitiveType.TRIANGLES):
                    m.d.sync += [
                        self.os_primitive.p.color.eq(self.is_vertex.p.color),
                    ]

        return m
