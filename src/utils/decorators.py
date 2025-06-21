import time
import functools
import io
import contextlib


def timing_decorator(func):
    """Decorator to measure execution time of methods and store within the instance."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        function_name = func.__name__

        if "_timing_stats" not in self.__dict__:
            self.__dict__["_timing_stats"] = {}

        if function_name in self.__dict__["_timing_stats"]:
            self.__dict__["_timing_stats"][function_name].append(elapsed_time)
        else:
            self.__dict__["_timing_stats"][function_name] = [elapsed_time]

        return result

    return wrapper


def print_avg_timings(instance):
    """Prints the average execution time of each function in the instance."""
    if "_timing_stats" not in instance.__dict__:
        print("No timing data found.")
        return

    print("\nAverage execution time per function:")
    for func, times in instance.__dict__["_timing_stats"].items():
        avg_time = sum(times) / len(times)
        print(f"{func}: {avg_time:.6f} seconds (called {len(times)} times)")

    print("\nMax execution time per function:")
    for func, times in instance._timing_stats.items():
        max_time = max(times)
        print(f"{func}: {max_time:.6f} seconds")


def mute_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)

    return wrapper
