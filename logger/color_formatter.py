"""The ColoredFormatter class."""

import logging
import os
import sys
import typing


# Returns escape codes from format codes
def esc(*codes: int) -> str:
    return "\033[" + ";".join(str(code) for code in codes) + "m"


escape_codes = {
    "reset": esc(0),
    "bold": esc(1),
    "thin": esc(2),
}

escape_codes_foreground = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "purple": 35,
    "cyan": 36,
    "white": 37,
    "light_black": 90,
    "light_red": 91,
    "light_green": 92,
    "light_yellow": 93,
    "light_blue": 94,
    "light_purple": 95,
    "light_cyan": 96,
    "light_white": 97,
}

escape_codes_background = {
    "black": 40,
    "red": 41,
    "green": 42,
    "yellow": 43,
    "blue": 44,
    "purple": 45,
    "cyan": 46,
    "white": 47,
    "light_black": 100,
    "light_red": 101,
    "light_green": 102,
    "light_yellow": 103,
    "light_blue": 104,
    "light_purple": 105,
    "light_cyan": 106,
    "light_white": 107,
    # Bold background colors don't exist,
    # but we used to provide these names.
    "bold_black": 100,
    "bold_red": 101,
    "bold_green": 102,
    "bold_yellow": 103,
    "bold_blue": 104,
    "bold_purple": 105,
    "bold_cyan": 106,
    "bold_white": 107,
}

# Foreground without prefix
for name, code in escape_codes_foreground.items():
    escape_codes["%s" % name] = esc(code)
    escape_codes["bold_%s" % name] = esc(1, code)
    escape_codes["thin_%s" % name] = esc(2, code)

# Foreground with fg_ prefix
for name, code in escape_codes_foreground.items():
    escape_codes["fg_%s" % name] = esc(code)
    escape_codes["fg_bold_%s" % name] = esc(1, code)
    escape_codes["fg_thin_%s" % name] = esc(2, code)

# Background with bg_ prefix
for name, code in escape_codes_background.items():
    escape_codes["bg_%s" % name] = esc(code)

# 256 colour support
for code in range(256):
    escape_codes["fg_%d" % code] = esc(38, 5, code)
    escape_codes["bg_%d" % code] = esc(48, 5, code)


def parse_colors(string: str) -> str:
    """Return escape codes from a color sequence string."""
    return "".join(escape_codes[n] for n in string.split(",") if n)


__all__ = (
    "default_log_colors",
    "ColoredFormatter",
    "LevelFormatter",
    "TTYColoredFormatter",
)

# Type aliases used in function signatures.
EscapeCodes = typing.Mapping[str, str]
LogColors = typing.Mapping[str, str]
SecondaryLogColors = typing.Mapping[str, LogColors]
if sys.version_info >= (3, 8):
    _FormatStyle = typing.Literal["%", "{", "$"]
else:
    _FormatStyle = str

# The default colors to use for the debug levels
default_log_colors = {
    "DEBUG": "light_cyan",
    "INFO": "thin_light_green",
    "WARNING": "bold_purple",
    "ERROR": "bold_red",
    "CRITICAL": "bold_red",
}

# The default format to use for each style
default_formats = {
    "%": "%(log_color)s%(levelname)s:%(name)s:%(message)s",
    "{": "{log_color}{levelname}:{name}:{message}",
    "$": "${log_color}${levelname}:${name}:${message}",
}


class ColoredRecord:
    """
    Wraps a LogRecord, adding escape codes to the internal dict.

    The internal dict is used when formatting the message (by the PercentStyle,
    StrFormatStyle, and StringTemplateStyle classes).
    """

    def __init__(self, record: logging.LogRecord, escapes: EscapeCodes) -> None:
        self.__dict__.update(record.__dict__)
        self.__dict__.update(escapes)


class ColoredFormatter(logging.Formatter):
    """
    A formatter that allows colors to be placed in the format string.

    Intended to help in creating more readable logging output.
    """

    def __init__(
        self,
        fmt: typing.Optional[str] = None,
        datefmt: typing.Optional[str] = None,
        style: _FormatStyle = "%",
        log_colors: typing.Optional[LogColors] = None,
        reset: bool = True,
        secondary_log_colors: typing.Optional[SecondaryLogColors] = None,
        validate: bool = True,
        stream: typing.Optional[typing.IO] = None,
        no_color: bool = False,
        force_color: bool = True,
        defaults: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    ) -> None:
        """
        Set the format and colors the ColoredFormatter will use.

        The ``fmt``, ``datefmt``, ``style``, and ``default`` args are passed on to the
        ``logging.Formatter`` constructor.

        The ``secondary_log_colors`` argument can be used to create additional
        ``log_color`` attributes. Each key in the dictionary will set
        ``{key}_log_color``, using the value to select from a different
        ``log_colors`` set.

        :Parameters:
        - fmt (str): The format string to use.
        - datefmt (str): A format string for the date.
        - log_colors (dict):
            A mapping of log level names to color names.
        - reset (bool):
            Implicitly append a color reset to all records unless False.
        - style ('%' or '{' or '$'):
            The format style to use.
        - secondary_log_colors (dict):
            Map secondary ``log_color`` attributes. (*New in version 2.6.*)
        - validate (bool)
            Validate the format string.
        - stream (typing.IO)
            The stream formatted messages will be printed to. Used to toggle colour
            on non-TTY outputs. Optional.
        - no_color (bool):
            Disable color output.
        - force_color (bool):
            Enable color output. Takes precedence over `no_color`.
        """

        # Select a default format if `fmt` is not provided.
        fmt = default_formats[style] if fmt is None else fmt

        if sys.version_info >= (3, 10):
            super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        elif sys.version_info >= (3, 8):
            super().__init__(fmt, datefmt, style, validate)
        else:
            super().__init__(fmt, datefmt, style)

        self.log_colors = log_colors if log_colors is not None else default_log_colors
        self.secondary_log_colors = (
            secondary_log_colors if secondary_log_colors is not None else {}
        )
        self.reset = reset
        self.stream = stream
        self.no_color = no_color
        self.force_color = force_color

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Format a message from a record object."""
        escapes = self._escape_code_map(record.levelname)
        wrapper = ColoredRecord(record, escapes)
        message = super().formatMessage(wrapper)  # type: ignore
        message = self._append_reset(message, escapes)
        return message

    def _escape_code_map(self, item: str) -> EscapeCodes:
        """
        Build a map of keys to escape codes for use in message formatting.

        If _blank_escape_codes() returns True, all values will be an empty string.
        """
        codes = {**escape_codes}
        codes.setdefault("log_color", self._get_escape_code(self.log_colors, item))
        for name, colors in self.secondary_log_colors.items():
            codes.setdefault("%s_log_color" % name, self._get_escape_code(colors, item))
        if self._blank_escape_codes():
            codes = {key: "" for key in codes.keys()}
        return codes

    def _blank_escape_codes(self):
        """Return True if we should be prevented from printing escape codes."""
        if self.force_color or "FORCE_COLOR" in os.environ:
            return False

        if self.no_color or "NO_COLOR" in os.environ:
            return True

        if self.stream is not None and not self.stream.isatty():
            return True

        return False

    @staticmethod
    def _get_escape_code(log_colors: LogColors, item: str) -> str:
        """Extract a color sequence from a mapping, and return escape codes."""
        return parse_colors(log_colors.get(item, ""))

    def _append_reset(self, message: str, escapes: EscapeCodes) -> str:
        """Add a reset code to the end of the message, if it's not already there."""
        reset_escape_code = escapes["reset"]

        if self.reset and not message.endswith(reset_escape_code):
            message += reset_escape_code

        return message


class LevelFormatter:
    """An extension of ColoredFormatter that uses per-level format strings."""

    def __init__(self, fmt: typing.Mapping[str, str], **kwargs: typing.Any) -> None:
        """
        Configure a ColoredFormatter with its own format string for each log level.

        Supports fmt as a dict. All other args are passed on to the
        ``colorlog.ColoredFormatter`` constructor.

        :Parameters:
        - fmt (dict):
            A mapping of log levels (represented as strings, e.g. 'WARNING') to
            format strings. (*New in version 2.7.0)
        (All other parameters are the same as in colorlog.ColoredFormatter)

        Example:

        formatter = colorlog.LevelFormatter(
            fmt={
                "DEBUG": "%(log_color)s%(message)s (%(module)s:%(lineno)d)",
                "INFO": "%(log_color)s%(message)s",
                "WARNING": "%(log_color)sWRN: %(message)s (%(module)s:%(lineno)d)",
                "ERROR": "%(log_color)sERR: %(message)s (%(module)s:%(lineno)d)",
                "CRITICAL": "%(log_color)sCRT: %(message)s (%(module)s:%(lineno)d)",
            }
        )
        """
        self.formatters = {
            level: ColoredFormatter(fmt=f, **kwargs) for level, f in fmt.items()
        }

    def format(self, record: logging.LogRecord) -> str:
        return self.formatters[record.levelname].format(record)


# Provided for backwards compatibility. The features provided by this subclass are now
# included directly in the `ColoredFormatter` class.
TTYColoredFormatter = ColoredFormatter
