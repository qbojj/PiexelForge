from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out
from amaranth_soc import csr

from ..utils.bus import internal_gpu_bus
from ..utils.layouts import VertexLayout, num_textures
from ..utils.types import IndexKind, InputTopology, address_shape
from .layouts import VertexInputAttributeDescription, VertexInputAttributes
from .types import index_shape

__all__ = [
    "IndexGenerator",
    "InputTopologyProcessor",
    "InputAssembly",
]


class IndexGenerator(wiring.Component):
    """Generates index stream based on index stream description register.

    Gets index stream description and outputs index stream.

    TODO: add memory burst support
    """

    index_stream: Out(stream.Signature(index_shape))

    bus: Out(internal_gpu_bus)
    csr_bus: In(csr.Signature(addr_width=4, data_width=8))

    ready: Out(1)

    class IndexStreamDescription(csr.Register, access="rw"):
        address: csr.Field(csr.action.RW, address_shape)
        count: csr.Field(csr.action.RW, index_shape)
        kind: csr.Field(csr.action.RW, IndexKind)

    def __init__(self):
        super().__init__()
        regs = csr.Builder(addr_width=4, data_width=8)
        self.config = regs.add("index_input", self.IndexStreamDescription())
        self.start = regs.add(
            "start", csr.Register(csr.Field(csr.action.W, unsigned(1)))
        )
        self.csr_bridge = csr.Bridge(regs.as_memory_map())
        self.csr_bus.memory_map = self.csr_bridge.memory_map

    def elaborate(self, platform) -> Module:
        m = Module()

        m.submodules += [self.csr_bridge]

        config = self.config.f

        address = Signal.like(config.address.data)
        kind = config.kind.data
        count = config.count.data

        index_increment = kind.matches(
            {
                IndexKind.U8: 1,
                IndexKind.U16: 2,
                IndexKind.U32: 4,
            }
        )
        index_shift = kind.matches(
            {
                IndexKind.U8: 0,
                IndexKind.U16: 1,
                IndexKind.U32: 2,
            }
        )

        data_read = Signal.like(self.master.dat_r)

        sel_width = len(self.master.sel)

        assert sel_width == 4  # 32-bit memory bus with byte granularity
        offset = address[0:2]
        extended_data = Cat(kind, offset).matches(
            {
                Cat(IndexKind.U8, 0): data_read[0:8],
                Cat(IndexKind.U8, 1): data_read[8:16],
                Cat(IndexKind.U8, 2): data_read[16:24],
                Cat(IndexKind.U8, 3): data_read[24:32],
                Cat(IndexKind.U16, 0): data_read[0:16],
                Cat(IndexKind.U16, 2): data_read[16:32],
                Cat(IndexKind.U32, 0): data_read[0:32],
            }
        )

        cur_idx = Signal.like(count)

        with m.If(self.index_stream.ready):
            m.d.sync += self.index_stream.valid.eq(0)

        m.d.comb += self.ready.f.r_data.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.ready.eq(1)
                with m.If(self.start.f.w_data & self.start.f.w_stb):
                    m.d.sync += [
                        cur_idx.eq(0),
                        address.eq(self.config.f.address.data),
                    ]
                    with m.If(count == 0):
                        m.next = "IDLE"
                    with m.Elif(kind == IndexKind.NOT_INDEXED):
                        m.next = "STREAM_NON_INDEXED"
                    with m.Else():
                        m.next = "MEM_READ_INIT"

            with m.State("STREAM_NON_INDEXED"):
                with m.If(~self.index_stream.valid | self.index_stream.ready):
                    m.d.sync += [
                        self.index_stream.payload.eq(cur_idx),
                        self.index_stream.valid.eq(1),
                        cur_idx.eq(cur_idx + 1),
                    ]
                    with m.If(cur_idx + 1 == count):  # last index streamed
                        m.next = "WAIT_FLUSH"

            with m.State("MEM_READ_INIT"):
                # initiate memory read

                bytes_remaining = (count - cur_idx) << index_shift
                max_mask = Signal.like(self.master.sel)
                with m.Switch(bytes_remaining):
                    for i in range(sel_width):
                        with m.Case(i):
                            m.d.comb += max_mask.eq((1 << i) - 1)
                    with m.Default():
                        m.d.comb += max_mask.eq(~0)

                m.d.sync += [
                    self.master.cyc.eq(1),
                    self.master.adr.eq(address),
                    self.master.we.eq(0),
                    self.master.stb.eq(1),
                    self.master.sel.eq(max_mask << offset),
                ]
                m.next = "MEM_READ_WAIT"
            with m.State("MEM_READ_WAIT"):
                # wait for ack and our index to be empty
                with m.If(self.master.ack):
                    m.d.sync += [
                        # deassert memory access
                        self.master.cyc.eq(0),
                        self.master.stb.eq(0),
                        data_read.eq(extended_data),
                    ]
                    m.next = "INDEX_SEND"
            with m.State("INDEX_SEND"):
                with m.If(~self.index_stream.valid | self.index_stream.ready):
                    next_addr = address + index_increment
                    m.d.sync += [
                        self.index_stream.payload.eq(extended_data),
                        self.index_stream.valid.eq(1),
                        address.eq(next_addr),
                        cur_idx.eq(cur_idx + 1),
                    ]
                    with m.If(cur_idx + 1 == count):
                        m.next = "WAIT_FLUSH"
                    with m.Elif(next_addr[0:offset_bits] != 0):
                        m.next = "INDEX_SEND"
                    with m.Else():
                        m.next = (
                            "MEM_READ_INIT"  # crossed word boundary -> read next word
                        )
            with m.State("WAIT_FLUSH"):
                with m.If(~self.index_stream.valid | self.index_stream.ready):
                    m.next = "IDLE"

        return m


class InputTopologyProcessor(wiring.Component):
    """Processes input topology description.

    Gets input index stream and outputs vertex index stream based on input topology.
    """

    index_stream: In(stream.Signature(index_shape))
    index_stream_output: Out(stream.Signature(index_shape))

    csr_bus: In(csr.Signature(addr_width=4, data_width=8))

    ready: Out(1)

    class InputTopologyDescription(csr.Register, access="rw"):
        input_topology: csr.Field(csr.action.RW, InputTopology)
        primitive_restart_enable: csr.Field(csr.action.RW, unsigned(1))
        primitive_restart_index: csr.Field(csr.action.RW, index_shape)

    def __init__(self):
        super().__init__()
        regs = csr.Builder(addr_width=4, data_width=8)
        self.config = regs.add("topology", self.InputTopologyDescription())
        self.csr_bridge = csr.Bridge(regs.as_memory_map())
        self.csr_bus.memory_map = self.csr_bridge.memory_map

    def elaborate(self, platform) -> Module:
        m = Module()

        m.submodules += [self.csr_bridge]

        # values for triangle strips/fans and line strips
        v1 = Signal(index_shape)
        v2 = Signal(index_shape)
        vertex_count = Signal(3)

        # max 3 output indices per input index
        max_amplification = 3
        to_send = Signal(data.ArrayLayout(index_shape, max_amplification))
        to_send_left = Signal(2)

        ready_for_input = Signal()

        m.d.comb += self.ready.eq(
            ready_for_input & ~self.index_stream.valid & ~self.index_stream_output.valid
        )
        m.d.comb += ready_for_input.eq(to_send_left == 0)

        with m.If(self.index_stream_output.ready):
            with m.Switch(to_send_left):
                for i in range(1, max_amplification + 1):
                    with m.Case(i):
                        m.d.sync += [
                            self.index_stream_output.payload.eq(to_send[i - 1]),
                            self.index_stream_output.valid.eq(1),
                            to_send_left.eq(to_send_left - 1),
                        ]

                        if i == 1:
                            # last one being sent -> we are ready for new input
                            m.d.comb += ready_for_input.eq(1)
                with m.Default():
                    m.d.sync += self.index_stream_output.valid.eq(0)

        with m.If(self.index_stream.valid & ready_for_input):
            m.d.comb += self.index_stream.ready.eq(1)
            idx = self.index_stream.payload

            with m.If(
                self.config.f.primitive_restart_enable.data
                & (idx == self.config.f.primitive_restart_index.data)
            ):
                m.d.sync += vertex_count.eq(0)  # reset on primitive restart
            with m.Else():
                with m.Switch(self.config.f.input_topology.data):
                    with m.Case(
                        InputTopology.POINT_LIST,
                        InputTopology.LINE_LIST,
                        InputTopology.TRIANGLE_LIST,
                    ):
                        m.d.sync += [
                            self.index_stream_output.p.eq(idx),
                            self.index_stream_output.valid.eq(1),
                        ]
                    with m.Case(InputTopology.LINE_STRIP):
                        with m.If(vertex_count == 0):
                            m.d.sync += [
                                v1.eq(idx),
                                vertex_count.eq(1),
                            ]
                        with m.Else():
                            m.d.sync += [
                                to_send[1].eq(v1),
                                to_send[0].eq(idx),
                                to_send_left.eq(2),
                                v1.eq(idx),
                            ]
                    with m.Case(InputTopology.TRIANGLE_STRIP):
                        with m.Switch(vertex_count):
                            with m.Case(0):
                                m.d.sync += [
                                    v1.eq(idx),
                                    vertex_count.eq(1),
                                    self.index_stream.ready.eq(1),
                                ]
                            with m.Case(1):
                                m.d.sync += [
                                    v2.eq(idx),
                                    vertex_count.eq(2),
                                    self.index_stream.ready.eq(1),
                                ]
                            with m.Case(2):
                                # Odd triangle -> indexes n, n+1, n+2
                                # so v1, v2, idx
                                m.d.sync += [
                                    to_send[2].eq(v1),
                                    to_send[1].eq(v2),
                                    to_send[0].eq(idx),
                                    to_send_left.eq(3),
                                    vertex_count.eq(3),
                                    v1.eq(v2),
                                    v2.eq(idx),
                                ]
                            with m.Case(3):
                                # Even triangle -> indexes n+1, n, n+2
                                # so v2, v1, idx
                                m.d.sync += [
                                    to_send[2].eq(v2),
                                    to_send[1].eq(v1),
                                    to_send[0].eq(idx),
                                    to_send_left.eq(3),
                                    v1.eq(v2),
                                    v2.eq(idx),
                                ]
                    with m.Case(InputTopology.TRIANGLE_FAN):
                        with m.Switch(vertex_count):
                            with m.Case(0):
                                m.d.sync += [
                                    v1.eq(idx),  # center vertex
                                    vertex_count.eq(1),
                                ]
                            with m.Case(1):
                                m.d.sync += [
                                    v2.eq(idx),  # first outer vertex
                                    vertex_count.eq(2),
                                ]
                            with m.Default():
                                m.d.sync += [
                                    to_send[2].eq(v1),  # center vertex
                                    to_send[1].eq(v2),  # previous outer vertex
                                    to_send[0].eq(idx),  # current outer vertex
                                    to_send_left.eq(3),
                                    v2.eq(idx),
                                ]
                    with m.Default():
                        m.d.sync += Assert(
                            0, "unsupported topology"
                        )  # unsupported topology

        return m


class InputAssembly(wiring.Component):
    """Input Assembly stage.

    Gets index stream and outputs vertex attribute stream.

    Also exposes following registers:
    - vertex_input_attributes: VertexInputAttributes - information about vertex attributes

    TODO: support other formats than Fixed 16.16 x number of components
    """

    index_stream: In(stream.Signature(index_shape))
    vertex_stream: Out(stream.Signature(VertexLayout))

    bus: Out(internal_gpu_bus)
    csr_bus: In(csr.Signature(addr_width=4, data_width=8))

    ready: Out(1)

    class VertexInputAttributes(csr.Register, access="rw"):
        def __init__(self):
            super().__init__(
                {
                    "position": csr.Field(
                        csr.action.RW, VertexInputAttributeDescription
                    ),
                    "normal": csr.Field(csr.action.RW, VertexInputAttributeDescription),
                    "texcoords": [
                        csr.Field(csr.action.RW, VertexInputAttributeDescription)
                        for _ in range(num_textures)
                    ],
                    "color": csr.Field(csr.action.RW, VertexInputAttributeDescription),
                }
            )

    def __init__(self):
        super().__init__()
        regs = csr.Builder(addr_width=4, data_width=8)
        self.vertex_input_attributes = regs.add(
            "vertex_input_attributes",
            csr.Register(csr.Field(csr.action.RW, VertexInputAttributes())),
        )
        self.csr_bridge = csr.Br2idge(regs.as_memory_map())
        self.csr_bus.memory_map = self.csr_bridge.memory_map

    def elaborate(self, platform) -> Module:
        m = Module()

        m.submodules += [self.csr_bridge]

        # fetch vertex at given index based on vertex input attributes

        idx = Signal.like(self.index_stream.payload)
        vtx = Signal.like(self.vertex_stream.payload)

        output_next_free = ~self.vertex_stream.valid | self.vertex_stream.ready

        with m.If(self.vertex_stream.ready):
            m.d.sync += self.vertex_stream.valid.eq(0)

        class AttrInfo:
            name: str
            desc: VertexInputAttributeDescription
            data_v: Signal

            @property
            def components(self):
                return self.data_v.shape().num_components.data

        attr_info = [
            AttrInfo(
                name="position",
                desc=self.vertex_input_attributes.f.position.data,
                data_v=vtx.position,
            ),
            AttrInfo(
                name="normal",
                desc=self.vertex_input_attributes.f.normal.data,
                data_v=vtx.normal,
            ),
            *[
                AttrInfo(
                    name=f"texcoord_{i}",
                    desc=self.vertex_input_attributes.f.texcoords[i].data,
                    data_v=vtx.texcoords[i],
                )
                for i in range(num_textures)
            ],
            AttrInfo(
                name="color",
                desc=self.vertex_input_attributes.f.color.data,
                data_v=vtx.color,
            ),
        ]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += [
                    self.ready.eq(~self.vertex_stream.valid & ~self.index_stream.valid),
                    self.index_stream.ready.eq(1),
                ]

                with m.If(self.index_stream.valid):
                    m.d.sync += idx.eq(self.index_stream.payload)
                    m.next = "FETCH_ATTR_0_START"

            for attr_no, attr in enumerate(attr_info):
                base_name = f"FETCH_ATTR_{attr_no}"
                with m.State(f"{base_name}_START"):
                    desc = attr["desc"]
                    with m.If(
                        desc.input_mode
                        == VertexInputAttributeDescription.InputMode.CONSTANT
                    ):
                        # constant value
                        m.d.sync += [
                            attr.data_v[i].eq(desc.constant_value[i])
                            for i in range(attr.components)
                        ]
                        m.next = f"{base_name}_DONE"
                    with m.Else():
                        # per-vertex attribute
                        addr = Signal.like(desc.per_vertex.address)
                        stride = Signal.like(desc.per_vertex.stride)

                        m.d.sync += addr.eq(desc.per_vertex.address + idx * stride)
                        m.next = f"{base_name}_MEM_READ_COMPONENT_0"

                for i in range(attr.components):
                    with m.State(f"{base_name}_MEM_READ_COMPONENT_{i}"):
                        # initiate memory read
                        m.d.sync += [
                            self.bus.cyc.eq(1),
                            self.bus.adr.eq(addr),
                            self.bus.we.eq(0),
                            self.bus.stb.eq(1),
                            self.bus.sel.eq(0b1111),  # 32-bits
                        ]
                        m.next = f"{base_name}_MEM_WAIT_{i}"

                    with m.State(f"{base_name}_MEM_WAIT_{i}"):
                        with m.If(self.bus.ack):
                            # parse and store
                            m.d.sync += attr.data_v[i].eq(
                                attr.data_v.shape().change_radix(self.bus.dat_r)
                            )

                            # deassert memory access
                            m.d.sync += [
                                self.bus.cyc.eq(0),
                                self.bus.stb.eq(0),
                            ]

                            if i + 1 < attr.components:
                                # next component

                                # 4 bytes per component (Fixed point 16.16)
                                m.d.sync += addr.eq(addr + 4)
                                m.next = f"{base_name}_MEM_READ_COMPONENT_{i + 1}"
                            else:
                                # all components read
                                m.next = f"{base_name}_DONE"

                    with m.State(f"{base_name}_DONE"):
                        if attr_no == len(attr_info) - 1:
                            # last attribute -> output vertex
                            with m.If(output_next_free):
                                m.d.sync += [
                                    self.vertex_stream.payload.eq(vtx),
                                    self.vertex_stream.valid.eq(1),
                                ]
                                m.next = "IDLE"
                        else:
                            m.next = f"FETCH_ATTR_{attr_no + 1}_START"

        return m
