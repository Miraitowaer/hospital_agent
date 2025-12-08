import json
from datetime import datetime, timedelta
from typing import Union

"redis://账号:密码@地址:6379/数据库index"

class MemCache(object):
    def __init__(
            self,
            redis_url,
            expire_time=None,
            key_prefix: str = "",
            val_decode=lambda val: json.loads(val.decode()),
    ):
        """
        :param rc: redis client
        :param expire_time: expire time (secs) in redis (>0), or None
        :param key_prefix: key_prefix_name
        :param val_decode: value decode function after fetch from redis

        :return: None
        """
        assert expire_time is None or expire_time > 0, "invalid expire time for redis"
        assert redis_url is not None, "redis_url is not available"
        self.redis_url = redis_url
        self.rc = self._init_client()
        self.expire_time = expire_time
        self.key_prefix = key_prefix
        self.key_prefix_len = len(self.key_prefix)
        self.val_decode = val_decode

    def _init_client(self):
        import redis
        return redis.StrictRedis(host='xxx', port=6379, db=0, decode_responses=True)
        # return redis.from_url(self.url)

    def _get_key(self, key):
        """key_prefix + key_id"""
        return self.key_prefix + str(key)

    def __setitem__(self, key, val):
        key = self._get_key(key)
        self.rc.set(key, val, ex=self.expire_time)

    def __getitem__(self, key):
        key = self._get_key(key)
        val = None
        if key:
            val = self.rc.get(key)
        if val and self.val_decode:
            try:
                val = self.val_decode(val)
            except:
                raise Exception("custom value decoding error for key [%s]" % str(key))
        return val

    def __contains__(self, key):
        key = self._get_key(key)
        # return self.rc.keys(key) != []
        return self.rc.exists(key)

    def keys(self, sub_key_prefix="*"):
        pattern = self._get_key(sub_key_prefix)
        keys = self.rc.keys(pattern)
        return [k.decode()[self.key_prefix_len:] for k in keys]

    def values(self):
        keys = self.keys()
        return [self[self._get_key(k)] for k in keys]

    def clear(self):
        pattern = self._get_key("*")
        keys = [k.decode() for k in self.rc.keys(pattern)]
        if not keys:
            return
        self.rc.delete(*keys)

    def close(self):
        self.rc.close()

    def batch_set(self, data, no_expire=False):
        """
        data: dict, key-val pair
        no_expire: no expire time flag, False default, use the self.expire_time
        """
        assert isinstance(data, dict), "batch set input must be a dict map"
        pipe = self.rc.pipeline()
        expire_time = self.expire_time
        if no_expire:
            expire_time = None
        for k, v in data.items():
            pipe.set(self._get_key(k), v, ex=expire_time)
        pipe.execute()

    def batch_get(self, sub_key_prefix="*"):
        keys = self.keys(sub_key_prefix)
        return {k: self[k] for k in keys}

    def exists(self, key):
        key = self._get_key(key)
        return self.rc.exists(key)

    def delete(self, key):
        key = self._get_key(key)
        self.rc.delete(key)

    def batch_extend_ttl(self, key):
        """批量延长符合30s过期的audio_key"""
        audio_key = self._get_key(key)
        ttl = self.rc.ttl(audio_key)
        if ttl <= 30:
            pipe = self.rc.pipeline()
            expire_at = datetime.now() + timedelta(seconds=self.expire_time)
            keys = self.rc.keys(f"*{key}*")
            for key in keys:
                pipe.expireat(key, expire_at)
            pipe.execute()

    def msetmx(self, data: dict):
        """
        :param: data: 包含key和value的字典数据
        """
        self.rc.msetnx(data)

    def list_add(self, key, data: Union[list, str], keep_length=10):
        """
        更新数据到list中
        :param key: redisKey
        :param data:
        :param keep_length: list最大长度
        :return: 成功插入数量
        """
        tmp_key, key = key, self._get_key(key)
        if isinstance(data, str):
            data = [data]
        len_of_key = self.list_len(tmp_key)
        pop_count = len_of_key + len(data) - keep_length
        if pop_count > 0:
            self.list_pop(tmp_key, pop_count)
        success_count = self.rc.lpush(key, *data)
        if self.expire_time:
            self.rc.expire(key, self.expire_time)
        return success_count

    def list_pop(self, key, count=1):
        """
        从列表中弹出指定count个数的元素
        """
        key = self._get_key(key)
        for index in range(count):
            self.rc.rpop(key)

    def list_len(self, key: str):
        """
        获取列表长度
        """
        key = self._get_key(key)
        return self.rc.llen(key)

    def list_get(self, key, start, stop):
        """
        获取列表指定下标的所有元素
        """
        key = self._get_key(key)
        return self.rc.lrange(key, start, stop)

    def list_get_all(self, key):
        """获取列表所有元素"""
        return self.list_get(key, 0, -1)

    def list_get_latest(self, key, latest_no: int):
        """获取列表最近的几个元素"""
        stop = latest_no - 1
        return self.list_get(key, 0, stop)

    def expire(self,key,expire_time):
        self.rc.expire(key, expire_time)

if __name__ == "__main__":
    import redis

    # url = "redis://:StevenWade2468@127.0.0.1:6379/0"
    # client = redis.from_url(url)

    # client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    data = [1, 2, 3, 4, 5, 6, 7]
    # data = [8,9,10,11,12]
    cache = MemCache("")
    key = "console_user_001"
    # cache.list_add(key, data)
    # print(cache.list_pop(3))
    print(cache.list_get_all(key))
    cache.expire(key,100)
    # print(cache.list_get_latest(key, 5))
