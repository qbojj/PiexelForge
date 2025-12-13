from amaranth import *
from amaranth.lib.wiring import Component, Out
from amaranth_soc.wishbone.bus import Interface, Signature
from transactron import *
from transactron.lib import Forwarder


class MemorySystem(Component):
    """

    Attributes
    ----------
    wb_bus: Out(Signature)
        Wishbone bus interface to read from memory.

    request_read: Method
        Method to request a memory read.
    read_resolve: Method
        Method to get the memory read response.

    request_write: Method
        Method to request a memory write.
    """

    wb_bus: Interface
    request_read: Method
    read_resolve: Method
    request_write: Method

    def __init__(self, wb_bus: Signature):
        super().__init__({"wb_bus": Out(wb_bus)})

        self.request_read = Method(
            i=[
                ("addr", wb_bus.addr_width),
                ("mask", self.wb_bus.sel.shape()),
            ]
        )
        self.read_resolve = Method(o=[("data", wb_bus.data_width)])
        self.request_write = Method(
            i=[
                ("addr", wb_bus.addr_width),
                ("mask", self.wb_bus.sel.shape()),
                ("data", wb_bus.data_width),
            ]
        )

        self.request_read.add_conflict(self.request_write)

    def elaborate(self, platform) -> TModule:
        m = TModule()

        busy = Signal(1)
        m.d.comb += busy.eq(self.wb_bus.cyc)

        m.submodules.read_fwd = read_fwd = Forwarder(self.wb_bus.data_width)

        with m.If(busy & self.wb_bus.ack):
            m.d.sync += [
                self.wb_bus.cyc.eq(0),
                self.wb_bus.stb.eq(0),
            ]

            with Transaction().body(m, ready=~self.wb_bus.we):
                read_fwd.write(m, self.wb_bus.dat_r)

        @def_method(m, self.request_read, ready=(~busy & ~read_fwd.read.ready))
        def _(addr, mask):
            m.d.sync += [
                self.wb_bus.adr.eq(addr),
                self.wb_bus.sel.eq(mask),
                self.wb_bus.we.eq(0),
                self.wb_bus.cyc.eq(1),
                self.wb_bus.stb.eq(1),
            ]

        @def_method(m, self.read_resolve)
        def _():
            return read_fwd.read(m)

        @def_method(m, self.request_write, ready=~busy)
        def _(addr, mask, data):
            m.d.sync += [
                self.wb_bus.adr.eq(addr),
                self.wb_bus.sel.eq(mask),
                self.wb_bus.dat_w.eq(data),
                self.wb_bus.we.eq(1),
                self.wb_bus.cyc.eq(1),
                self.wb_bus.stb.eq(1),
            ]

        return m
