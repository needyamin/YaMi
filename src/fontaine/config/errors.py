"""Configuration errors raised for invalid or ambiguous configuration."""


class ConfigError(ValueError):
    """Raised when configuration files, overrides, or merged settings are invalid.

    The message always contains the dotted path of the offending key so the
    user can locate the problem in their YAML file directly.
    """
