def maybe_detector(name, tags):
    try:
        from giskard.scanner.decorators import detector

        return detector(name, tags)
    except (ImportError, ModuleNotFoundError):

        def inner(cls):
            return cls

        return inner
