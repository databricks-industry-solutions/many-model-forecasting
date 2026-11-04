"""intervals.spark.Materializer: persist() where available, temporary tables where it is rejected
(serverless clients before environment version 5: CACHE_REQUIRES_NEWER_CLIENT_ON_SERVERLESS_GC)."""
from mmf_sa.intervals import spark as cs


class _Writer:
    def __init__(self, spark):
        self.spark = spark

    def mode(self, mode):
        return self

    def saveAsTable(self, name):
        self.spark.tables[name] = "data"


class _NoCacheFrame:
    def __init__(self, spark):
        self.write = _Writer(spark)

    def persist(self):
        raise RuntimeError("[CACHE_REQUIRES_NEWER_CLIENT_ON_SERVERLESS_GC] df.persist() requires a newer client")

    def unpersist(self):
        raise RuntimeError("[CACHE_REQUIRES_NEWER_CLIENT_ON_SERVERLESS_GC] df.unpersist() requires a newer client")


class _Frame:
    persisted = False

    def persist(self):
        self.persisted = True
        return self

    def unpersist(self):
        self.persisted = False


class _Spark:
    def __init__(self):
        self.tables, self.dropped = {}, []

    def table(self, name):
        return ("table", name)

    def sql(self, query):
        self.dropped.append(query.split()[-1])


def test_falls_back_to_temporary_tables(caplog):
    spark = _Spark()
    m = cs.Materializer(spark, "cat.db.eval_pi_tmp")
    with caplog.at_level("WARNING"):
        a = m.cache(_NoCacheFrame(spark))
        b = m.cache(_NoCacheFrame(spark))
    assert "temporary tables" in caplog.text
    assert a[0] == "table" and a[1].startswith("cat.db.eval_pi_tmp_") and a != b
    assert len(spark.tables) == 2
    m.uncache(a)
    assert spark.dropped == [a[1]]
    m.close()
    assert spark.dropped == [a[1], b[1]]


def test_persists_when_available():
    m = cs.Materializer(_Spark(), "x")
    df, other = _Frame(), _Frame()
    assert m.cache(df) is df and df.persisted
    m.cache(other)
    m.uncache(df)
    assert not df.persisted and other.persisted
    m.close()
    assert not other.persisted
