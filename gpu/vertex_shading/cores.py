from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from ..utils.layouts import PrimitiveAssemblyLayout, ShadingVertexLayout
from ..utils.types import Vector3_mem, Vector4_mem


class LightPropertyLayout(data.Struct):
    """Light properties layout"""

    position: Vector4_mem
    ambient: Vector3_mem
    diffuse: Vector3_mem
    specular: Vector3_mem


class MaterialPropertyLayout(data.Struct):
    """Material properties layout"""

    ambient: Vector3_mem
    diffuse: Vector3_mem
    specular: Vector3_mem
    shininess: unsigned(32)


class VertexShading(wiring.Component):
    """Vertex shading core

    Shades incoming vertices using Gouraud shading model.
    Outputs shaded vertices for rasterization stage.

    Input: ShadingVertexLayout
    Output: ShadingVertexLayout

    Uses following wires for material properties:
    - material_ambient: Ambient color of the material (vec3)
    - material_diffuse: Diffuse color of the material (vec3)
    - material_specular: Specular color of the material (vec3)
    - material_shininess: Shininess coefficient of the material (float)

    Uses following wires for light properties:
    - light: array of light property structures
    """

    is_vertex: In(stream.Signature(ShadingVertexLayout))
    os_vertex: Out(stream.Signature(PrimitiveAssemblyLayout))

    material: In(MaterialPropertyLayout)
    lights: In(data.ArrayLayout(LightPropertyLayout, 1))

    ready: Out(1)

    def __init__(self, num_lights=1):
        self.num_lights = num_lights
        super().__init__()

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
