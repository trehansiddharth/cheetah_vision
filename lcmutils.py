import lcm
import os
import time
import threading
import collections
import bisect

class History:
    def __init__(self, ts, values):
        self.capacity = ts.maxlen
        self.ts = ts
        self.values = values
    
    @staticmethod
    def with_capacity(capacity):
        return History(collections.deque(maxlen=capacity),
            collections.deque(maxlen=capacity))
    
    def update(self, t, value):
        if len(self) >= self.capacity:
            del self[0]
        i_at = bisect.bisect(self.ts, t)
        self.ts.insert(i_at, t)
        self.values.insert(i_at, value)
        return i_at

    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, slc):
        return History(self.ts[slc], self.values[slc])
    
    def __setitem__(self, slc, values):
        self.values[slc] = values
    
    def __delitem__(self, slc):
        del self.ts[slc]
        del self.values[slc]
    
    def lastest(self):
        return self.ts[-1], self.values[-1]
    
    def at(self, t_at):
        if len(self.ts) == 0:
            return None, None
        else:
            i_at = bisect.bisect(self.ts, t_at)
            i_at = min(i_at, len(self.ts) - 1)

            return self.ts[i_at], self.values[i_at]
    
    def index_of(self, t):
        i = bisect.bisect(self.ts, t)
        if i > 0 and self.ts[i-1] == t:
            return i - 1
        else:
            return None

def node(url):
    if url is None:
        if "LCM_DEFAULT_URL" in os.environ:
            url = os.environ["LCM_DEFAULT_URL"]
        else:
            url = "udpm://239.255.76.67:7667?ttl=255"
    return lcm.LCM(url)

def print_stats(channel, delays, interval):
    def runner(interval):
        while True:
            if len(delays) >= 2:
                frequency = len(delays) / (delays.ts[-1] - delays.ts[0])
                latency = sum(delays.values) / len(delays)
                print("[{}] {} Hz, -{} s".format(channel, frequency, latency))
                time.sleep(interval)
    print_stats_thread = threading.Thread(target=runner, args=(interval,))
    print_stats_thread.setDaemon(True)
    print_stats_thread.start()

def unsubscribe(node, subscriptions):
    for subscription in subscriptions:
        node.unsubscribe(subscription)

def subscribe(node, channel, lcm_type, callback, verbose=True):
    delays = History.with_capacity(100)
    def raw_callback(channel, data):
        t_start = time.time()
        callback(lcm_type.decode(data))
        t_end = time.time()
        if verbose:
            delays.update(t_start, t_end - t_start)
    subscription = node.subscribe(channel, raw_callback)
    subscription.set_queue_capacity(1)
    if verbose:
        print_stats(channel, delays, 1.0)
    return [subscription]

def subscribe_sync(node, channel_sync, lcm_type_sync,
    channels, lcm_types, callback, verbose=True, memory=100):
    histories = [History.with_capacity(memory) for channel in channels]
    delays = History.with_capacity(memory)
    def gen_callback(i):
        def raw_callback(channel, data):
            msg = lcm_types[i].decode(data)
            histories[i].update(msg.timestamp, msg)
        return raw_callback
    def sync_callback(channel, data):
        t_start = time.time()
        msg = lcm_type_sync.decode(data)
        t_sync = msg.timestamp
        other_msgs = [history.at(t_sync)[1] for history in histories]
        if not None in other_msgs:
            callback(msg, *other_msgs)
            t_end = time.time()
            if verbose:
                delays.update(t_start, t_end - t_start)
    subscriptions = []
    for i, channel in enumerate(channels):
        subscription = node.subscribe(channel, gen_callback(i))
        subscription.set_queue_capacity(1)
        subscriptions.append(subscription)
    subscription = node.subscribe(channel_sync, sync_callback)
    subscription.set_queue_capacity(1)
    subscriptions.insert(0, subscription)
    if verbose:
        print_stats(", ".join(channels) + " ~ " + channel_sync, delays, 1.0)
    return subscriptions

def subscribe_sync_exact(node, channels, lcm_types, callback, verbose=True, memory=10):
    history = History.with_capacity(memory)
    delays = History.with_capacity(memory)
    n = len(channels)
    def gen_callback(i):
        def raw_callback(channel, data):
            msg = lcm_types[i].decode(data)
            i_at = history.index_of(msg.timestamp)
            if i_at is None:
                value = [None] * n
                value[i] = msg
                i_at = history.update(msg.timestamp, value)
            elif i_at is not None:
                history.values[i_at][i] = msg
            if i_at is not None:
                if not None in history.values[i_at]:
                    callback(*history.values[i_at])
                i_before = 0
                for _ in range(i_at):
                    if history.values[i_before][i] is None:
                        del history[i_before]
                    else:
                        i_before += 1
        return raw_callback
    subscriptions = []
    for i, channel in enumerate(channels):
        subscription = node.subscribe(channel, gen_callback(i))
        subscription.set_queue_capacity(1)
        subscriptions.append(subscription)
    if verbose:
        print_stats(", ".join(channels), delays, 1.0)
    return subscriptions

def publish(node, channel, message):
    node.publish(channel, message.encode())
        