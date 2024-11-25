import py_trees


class StatusMonitor:
    route_completion = 0  # RouteCompletionTest.actual_value
    driven_distance = 0  # DrivenDistanceTest.actual_value
    terminal = False
    terminal_reason = None

    @classmethod
    def get_route_completion(cls, func):
        def wrapper(*args, **kwargs):
            cls.route_completion = func(*args, **kwargs)
            new_status = func(*args, **kwargs)

            if new_status == py_trees.common.Status.FAILURE:
                cls.terminal = True
                cls.terminal_reason = "RouteCompletionTest"

            cls.route_completion = getattr(args[0], "actual_value")
            cls.driven_distance = getattr(args[0], "actual_distance")

            return new_status

        return wrapper

    @classmethod
    def get_status(cls, func):
        criteria = func.__qualname__.split(".")[0]

        def wrapper(*args, **kwargs):
            new_status = func(*args, **kwargs)

            if new_status == py_trees.common.Status.FAILURE:
                cls.terminal = True
                cls.terminal_reason = criteria

            return new_status

        return wrapper

    @classmethod
    def reset(cls):
        cls.route_completion = 0
        cls.driven_distance = 0
        cls.terminal = False
        cls.terminal_reason = None
