from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.wiring import Out
from amaranth_soc.memory import MemoryMap
from amaranth_soc.wishbone.bus import Signature


def get_memory_resource(mmap: MemoryMap, name: str):
    subvalues = name.split(".")

    for res in mmap.all_resources():
        if all(
            res.path[i] == MemoryMap.Name(subvalues[i]) for i in range(len(subvalues))
        ):
            return res
    raise KeyError(f"Resource {name} not found in memory map")


class DebugAccess(wiring.Component):
    """Simple Wishbone master for testbenches.

    Can perform single read and write transactions.

    Members
    -------
    wb_bus : ``Out(wishbone.Signature(...))``
        Wishbone bus interface.
    """

    def __init__(self, addr_width, data_width, granularity=None):
        if granularity is None:
            granularity = data_width

        super().__init__(
            {
                "wb_bus": Out(
                    Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        granularity=granularity,
                    )
                )
            }
        )

        self._addr_width = addr_width
        self._data_width = data_width
        self._granularity = granularity

    @property
    def addr_width(self):
        return self._addr_width

    @property
    def data_width(self):
        return self._data_width

    @property
    def granularity(self):
        return self._granularity

    def elaborate(self, platform):
        return Module()

    async def read(self, ctx, addr: int, width: int) -> list[int]:
        """Perform a read transaction.

        Parameters
        ----------
        ctx : SimulatorProcess
            The simulation context.
        addr : int
            The address to read from.
        width : int
            The number of bytes to read.
        Returns
        -------
        list[int]
            The read data as a list of Consts, one per byte.
        """

        data = []
        for i in range(width):
            ctx.set(self.wb_bus.adr, addr + i)
            ctx.set(self.wb_bus.sel, ~0)
            ctx.set(self.wb_bus.cyc, 1)
            ctx.set(self.wb_bus.stb, 1)
            ctx.set(self.wb_bus.we, 0)
            await ctx.tick().until(self.wb_bus.ack)
            data.append(ctx.get(self.wb_bus.dat_r))
            ctx.set(self.wb_bus.cyc, 0)
            ctx.set(self.wb_bus.stb, 0)
            await ctx.tick()
        return data

    async def write(self, ctx, addr: int, data: list[int]) -> None:
        """Perform a write transaction.

        Parameters
        ----------
        ctx : SimulatorProcess
            The simulation context.
        addr : int
            The address to write to.
        data : list[int]
            The data to write as a list of Consts, one per byte.
        """

        for i, datum in enumerate(data):
            ctx.set(self.wb_bus.adr, addr // (self.data_width // self.granularity) + i)
            ctx.set(self.wb_bus.dat_w, datum)
            ctx.set(self.wb_bus.sel, ~0)
            ctx.set(self.wb_bus.cyc, 1)
            ctx.set(self.wb_bus.stb, 1)
            ctx.set(self.wb_bus.we, 1)
            await ctx.tick().until(self.wb_bus.ack)
            ctx.set(self.wb_bus.cyc, 0)
            ctx.set(self.wb_bus.stb, 0)
            await ctx.tick()
