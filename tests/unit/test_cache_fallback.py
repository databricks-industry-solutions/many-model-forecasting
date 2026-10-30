"""calibration_spark.cache / uncache fall back to no caching where persist() is rejected
(serverless clients before environment version 5: CACHE_REQUIRES_NEWER_CLIENT_ON_SERVERLESS_GC)."""
from mmf_sa import calibration_spark as cs


class _NoCacheFrame:
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


def test_cache_falls_back_without_caching(caplog):
    df = _NoCacheFrame()
    with caplog.at_level("WARNING"):
        assert cs.cache(df) is df
        cs.uncache(df)
    assert "caching is not available" in caplog.text


def test_cache_persists_when_available():
    df = _Frame()
    assert cs.cache(df) is df and df.persisted
    cs.uncache(df)
    assert not df.persisted
