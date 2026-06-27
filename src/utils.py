def SimpleTimeFormater(seconds, return_scale=False):
    if seconds < 60.0:
        if seconds < 1.0:
            if seconds < 0.001:
                return (
                    f"{int(seconds * 1000 * 1000)}µs"
                    if not return_scale
                    else (10**6, "µs")
                )
            else:
                return (
                    f"{int(seconds * 1000)}ms"
                    if not return_scale
                    else (1000, "ms")
                )
        else:
            return f"{seconds:.1f}s" if not return_scale else (1, "s")
    elif seconds < 3600:
        return (
            f"{int(seconds // 60)}m {int(seconds % 60)}s"
            if not return_scale
            else (1 / 60, "m")
        )
    else:
        return (
            f"{int(seconds // 3600)}h {(int(seconds / 60) % 60)}m"
            if not return_scale
            else (1 / 3600, "h")
        )
