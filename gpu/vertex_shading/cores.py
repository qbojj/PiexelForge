from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out
from amaranth_soc import csr

from ..utils.layouts import PrimitiveAssemblyLayout, ShadingVertexLayout
from ..utils.types import Vector3_mem, Vector4_mem


class VertexShading(wiring.Component):
    """Vertex shading core

    Shades incoming vertices using Gouraud shading model.
    Outputs shaded vertices for rasterization stage.

    Input: ShadingVertexLayout
    Output: ShadingVertexLayout

    Uses following registers for material properties:
    - material_ambient: Ambient color of the material (vec3)
    - material_diffuse: Diffuse color of the material (vec3)
    - material_specular: Specular color of the material (vec3)
    - material_shininess: Shininess coefficient of the material (float)

    Uses following registers for light properties:
    - light_position: Position of the light in world space (vec4)
    - light_ambient: Ambient color of the light (vec3)
    - light_diffuse: Diffuse color of the light (vec3)
    - light_specular: Specular color of the light (vec3)

    for NUM_LIGHTS lights.

    TODO: implement lighting (for now only passthrough)
    """

    is_vertex: In(stream.Signature(ShadingVertexLayout))
    os_vertex: Out(stream.Signature(PrimitiveAssemblyLayout))

    ready: Out(1)

    class LightReg(csr.Register, access="rw"):
        position: csr.Field(csr.action.RW, Vector4_mem)
        ambient: csr.Field(csr.action.RW, Vector3_mem)
        diffuse: csr.Field(csr.action.RW, Vector3_mem)
        specular: csr.Field(csr.action.RW, Vector3_mem)

    def __init__(self, num_lights):
        super().__init__()
        # TODO: implement

    def elaborate(self, platform):
        m = Module()

        m.d.comb += self.ready.eq(~self.os_vertex.valid)

        m.d.comb += [
            self.os_vertex.p.position_proj.eq(self.is_vertex.p.position_proj),
            self.os_vertex.p.texcoords.eq(self.is_vertex.p.texcoords),
            self.os_vertex.p.color.eq(self.is_vertex.p.color),
            self.os_vertex.p.color_back.eq(self.is_vertex.p.color),
            self.os_vertex.valid.eq(self.is_vertex.valid),
            self.is_vertex.ready.eq(self.os_vertex.ready),
        ]

        return m
