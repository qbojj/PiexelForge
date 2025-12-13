from amaranth import *
from amaranth.lib import stream, wiring
from transactron import *
from transactron.lib.connectors import ConnectTrans


def connect_stream_like_to_stream_like(
    m: TModule,
    source: Method | stream.Interface,
    sink: Method | stream.Interface,
) -> None:
    """Connects any Method or stream.Interface to any other Method or stream.Interface using adapters."""

    if isinstance(source, stream.Interface) and isinstance(sink, stream.Interface):
        wiring.connect(m, source, sink)
    elif isinstance(source, Method) and isinstance(sink, stream.Interface):
        with m.If(sink.ready):
            m.d.sync += sink.valid.eq(0)

        with Transaction().body(m, ready=(sink.ready | ~sink.valid)):
            m.d.sync += sink.payload.eq(source(m))
            m.d.sync += sink.valid.eq(1)
    elif isinstance(source, stream.Interface) and isinstance(sink, Method):
        with Transaction().body(m, ready=source.valid):
            sink(m, source.payload)
            m.d.comb += source.ready.eq(1)
    elif isinstance(source, Method) and isinstance(sink, Method):
        m.submodules += ConnectTrans.create(source, sink)
