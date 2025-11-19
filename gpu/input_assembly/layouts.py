from amaranth import *
from amaranth.lib import data, enum


class VertexInputAttributeDescription(data.Struct):
    class InputMode(enum.Enum, shape=1):
        CONSTANT = 0
        PER_VERTEX = 1

    input_mode: InputMode
    data: data.UnionLayout(
        {
            "per_vertex": data.StructLayout(
                {
                    "address": address_shape,
                    "stride": stride_shape,
                    # for now assume format (FixedPoint 16.16) for all attributes
                    # with appropriate number of components
                }
            ),
            "constant_value": Vector4,
        }
    )
