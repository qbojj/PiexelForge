from amaranth import *
from amaranth.lib import data

from .types import (
    FixedPoint,
    Vector2,
    Vector3,
    Vector4,
    address_shape,
    stride_shape,
    texture_coord_shape,
)

texture_coords = data.ArrayLayout(Vector2, 2)

# Max 4kx4k textures
num_textures = 2
texture_position = data.ArrayLayout(unsigned(texture_coord_shape), num_textures)


class VertexLayout(data.Struct):
    position: Vector4
    normal: Vector3
    texcoords: texture_coords
    color: Vector4


class ShadingVertexLayout(data.Struct):
    position_world: Vector4
    position_view: Vector4  # In homogeneous coordinates after w-divide
    normal_world: Vector3
    texcoords: texture_coords  # After transforms
    color: Vector4


class PrimitiveAssemblyLayout(data.Struct):
    position_view: Vector4
    texcoords: texture_coords
    color: Vector4
    color_back: Vector4


class FragmentLayout(data.Struct):
    depth: FixedPoint
    texcoords: texture_coords
    color: Vector4  # rgba in linear space
    coord_pos: texture_position


class FramebufferInfoLayout(data.Struct):
    width: texture_coord_shape
    height: texture_coord_shape

    color_address: address_shape  # assume R8G8B8A8
    color_pitch: stride_shape  # in bytes
    depth_address: address_shape  # assume D16
    depth_pitch: stride_shape  # in bytes
    stencil_address: address_shape  # S8
    stencil_pitch: stride_shape  # in bytes
