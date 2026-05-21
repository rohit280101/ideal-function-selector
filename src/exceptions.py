class AppError(Exception):
    pass

class DataLoadError(AppError):
    pass

class SchemaMismatchError(AppError):
    pass

class NoIdealFunctionError(AppError):
    pass
