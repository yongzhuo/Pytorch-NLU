# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/8/21 21:53
# @author  : Mo
# @function: tqdm(version=v4.62.1), no threading, no multiprocessing
# @url     : https://github.com/tqdm/tqdm


from platform import system as _curos
import time
import sys
import re
import os
try:
    from weakref import WeakSet
    _unicode = unicode
    _unich = unichr
    _range = xrange
except Exception as e:
    _range = range
    _unicode = str
    WeakSet = set
    _unich = chr

CUR_OS = _curos()
IS_WIN = CUR_OS in ['Windows', 'cli']
IS_NIX = (not IS_WIN) and any(
    CUR_OS.startswith(i) for i in
    ['CYGWIN', 'MSYS', 'Linux', 'Darwin', 'SunOS',
     'FreeBSD', 'NetBSD', 'OpenBSD'])
try:
    if IS_WIN:
        import colorama
        colorama.init()
    else:
        colorama = None
except ImportError:
    colorama = None

UTF_FMT = u" " + u''.join(map(_unich, range(0x258F, 0x2587, -1)))
RE_ANSI = re.compile(r"\x1b\[[;\d]*[A-Za-z]")
ASCII_FMT = " 123456789#"


class tqdm:
    """
    url: https://github.com/tqdm/tqdm
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.
    """
    def __init__(self, iterable=None, desc=None, unit_scale=False, unit_divisor=1000, gui=False):
        total = len(iterable)
        file = sys.stderr
        # Store the arguments
        self.iterable = iterable
        self.desc = desc or ''
        self.total = total
        self.ascii = ascii
        self.fp = file
        self.dynamic_miniters = True
        self.dynamic_ncols = False
        self.disable = False
        self.unit_divisor = unit_divisor
        self.unit_scale = unit_scale
        from time import time
        self._time = time
        self.unit = 'it'
        self.gui = gui
        self.bar_format = None
        self.avg_time = None
        self.postfix = None
        self.ncols = None
        # Init the iterations counters
        self.last_print_n = 0
        self.mininterval = 0.1
        self.smoothing = 0.3
        self.miniters = 0
        self.pos = 0
        self.n = 0

        if not gui:
            # Initialize the screen printer
            self.sp = self.status_printer(self.fp)
            self.display()

        # Init the time counter
        self.last_print_t = self._time()
        # NB: Avoid race conditions by setting start_t at the very end of init
        self.start_t = self.last_print_t

    def __repr__(self):
        return self.format_meter(**self.format_dict)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        mininterval = self.mininterval
        miniters = self.miniters
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        n = self.n
        smoothing = self.smoothing
        avg_time = self.avg_time
        _time = self._time

        for obj in iterable:
            yield obj
            # Update and possibly print the progressbar.
            # Note: does not call self.update(1) for speed optimisation.
            n += 1
            # check counter first to avoid calls to time()
            if n - last_print_n >= self.miniters:
                miniters = self.miniters  # watch monitoring thread changes
                delta_t = _time() - last_print_t
                if delta_t >= mininterval:
                    cur_t = _time()
                    delta_it = n - last_print_n
                    # EMA (not just overall average)
                    if smoothing and delta_t and delta_it:
                        rate = delta_t / delta_it
                        avg_time = self.ema(rate, avg_time, smoothing)
                        self.avg_time = avg_time

                    self.n = n
                    self.display()
                    # Store old values for next call
                    self.n = self.last_print_n = last_print_n = n
                    self.last_print_t = last_print_t = cur_t
                    self.miniters = miniters

        # Closing the progress bar.
        # Update some internal variables for close().
        self.last_print_n = last_print_n
        self.n = n
        self.miniters = miniters

    def display(self, msg=None, pos=None):
        """
        Use `self.sp` and to display `msg` in the specified `pos`.

        Parameters
        ----------
        msg  : what to display (default: repr(self))
        pos  : position to display in. (default: abs(self.pos))
        """
        if pos is None:
            pos = abs(self.pos)
        if pos:
            self.moveto(pos)
        self.sp(self.__repr__() if msg is None else msg)
        if pos:
            self.moveto(-pos)

    def moveto(self, n):
        """
        ANSI序列以 'ESC字符'+'[' 起始(在纯DOS下双击'ESC'键可获得'ESC字符'*1) 
        在Python中'ESC字符'可以用'\x1b'来表示 在这之后接具体的控制码即可 是不是特别方便(部分代码如下)
        \x1b[nA    光标上移
        \x1b[nB    光标下移
        \x1b[nC    光标右移
        \x1b[nD    光标左移(n 为行数/字符数)
        \x1b[2J    清屏(把2换成其他数字会有不同的清屏效果)
        \x1b[x;yH  调整屏幕坐标(x,y的单位是字符)
        \x1b?25l   隐藏光标
        \x1b?25h   显示光标
        """
        def _term_move_up():  # pragma: no cover
            return '' if (os.name == 'nt') and (colorama is None) else '\x1b[A'
        self.fp.write(_unicode('\n' * n + _term_move_up() * -n))
        self.fp.flush()

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', unit='it', unit_scale=False, rate=None,
                     bar_format=None, postfix=None, unit_divisor=1000, **extra_kwargs):
        """
        Return a string-based progress bar given some parameters

        Parameters
        ----------
        n  : int
            Number of finished iterations.
        total  : int
            The expected total number of iterations. If meaningless (), only
            basic progress statistics are displayed (no ETA).
        elapsed  : float
            Number of seconds passed since start.
        ncols  : int, optional
            The width of the entire output message. If specified,
            dynamically resizes the progress meter to stay within this bound
            [default: None]. The fallback meter width is 10 for the progress
            bar + no limit for the iterations counter and statistics. If 0,
            will not print any meter (only stats).
        prefix  : str, optional
            Prefix message (included in total width) [default: ''].
            Use as {desc} in bar_format string.
        ascii  : bool, optional or str, optional
            If not set, use unicode (smooth blocks) to fill the meter
            [default: False]. The fallback is to use ASCII characters
            " 123456789#".
        unit  : str, optional
            The iteration unit [default: 'it'].
        unit_scale  : bool or int or float, optional
            If 1 or True, the number of iterations will be printed with an
            appropriate SI metric prefix (k = 10^3, M = 10^6, etc.)
            [default: False]. If any other non-zero number, will scale
            `total` and `n`.
        rate  : float, optional
            Manual override for iteration rate.
            If [default: None], uses n/elapsed.
        bar_format  : str, optional
            Specify a custom bar string formatting. May impact performance.
            [default: '{l_bar}{bar}{r_bar}'], where
            l_bar='{desc}: {percentage:3.0f}%|' and
            r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
              '{rate_fmt}{postfix}]'
            Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
              percentage, rate, rate_fmt, rate_noinv, rate_noinv_fmt,
              rate_inv, rate_inv_fmt, elapsed, elapsed_s,
              remaining, remaining_s, desc, postfix, unit.
            Note that a trailing ": " is automatically removed after {desc}
            if the latter is empty.
        postfix  : *, optional
            Similar to `prefix`, but placed at the end
            (e.g. for additional stats).
            Note: postfix is usually a string (not a dict) for this method,
            and will if possible be set to postfix = ', ' + postfix.
            However other types are supported (#382).
        unit_divisor  : float, optional
            [default: 1000], ignored unless `unit_scale` is True.

        Returns
        -------
        out  : Formatted meter and stats, ready to display.
        """

        # sanity check: total
        if total and n > total:
            total = None

        # apply custom scale if necessary
        if unit_scale and unit_scale not in (True, 1):
            if total:
                total *= unit_scale
            n *= unit_scale
            if rate:
                rate *= unit_scale  # by default rate = 1 / self.avg_time
            unit_scale = False

        elapsed_str = tqdm.format_interval(elapsed)

        # if unspecified, attempt to use rate = average speed
        # (we allow manual override since predicting time is an arcane art)
        if rate is None and elapsed:
            rate = n / elapsed
        inv_rate = 1 / rate if rate else None
        format_sizeof = tqdm.format_sizeof
        rate_noinv_fmt = ((format_sizeof(rate) if unit_scale else
                           '{0:5.2f}'.format(rate))
                          if rate else '?') + unit + '/s'
        rate_inv_fmt = ((format_sizeof(inv_rate) if unit_scale else
                         '{0:5.2f}'.format(inv_rate))
                        if inv_rate else '?') + 's/' + unit
        rate_fmt = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt

        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = format_sizeof(total, divisor=unit_divisor) \
                if total is not None else '?'
        else:
            n_fmt = str(n)
            total_fmt = str(total) if total is not None else '?'

        try:
            postfix = ', ' + postfix if postfix else ''
        except TypeError:
            pass

        remaining = (total - n) / rate if rate and total else 0
        remaining_str = tqdm.format_interval(remaining) if rate else '?'

        # format the stats displayed to the left and right sides of the bar
        if prefix:
            # old prefix setup work around
            bool_prefix_colon_already = (prefix[-2:] == ": ")
            l_bar = prefix if bool_prefix_colon_already else prefix + ": "
        else:
            l_bar = ''

        r_bar = '| {0}/{1} [{2}<{3}, {4}{5}]'.format(
            n_fmt, total_fmt, elapsed_str, remaining_str, rate_fmt, postfix)

        # Custom bar formatting
        # Populate a dict with all available progress indicators
        format_dict = dict(
            n=n, n_fmt=n_fmt, total=total, total_fmt=total_fmt,
            rate=inv_rate if inv_rate and inv_rate > 1 else rate,
            rate_fmt=rate_fmt, rate_noinv=rate,
            rate_noinv_fmt=rate_noinv_fmt, rate_inv=inv_rate,
            rate_inv_fmt=rate_inv_fmt,
            elapsed=elapsed_str, elapsed_s=elapsed,
            remaining=remaining_str, remaining_s=remaining,
            l_bar=l_bar, r_bar=r_bar,
            desc=prefix or '', postfix=postfix, unit=unit,
            # bar=full_bar,  # replaced by procedure below
            **extra_kwargs)

        # total is known: we can predict some stats
        if total:
            # fractional and percentage progress
            frac = n / total
            percentage = frac * 100

            l_bar += '{0:3.0f}%|'.format(percentage)

            if ncols == 0:
                return l_bar[:-1] + r_bar[1:]

            if bar_format:
                format_dict.update(l_bar=l_bar, percentage=percentage)
                # , bar=full_bar  # replaced by procedure below

                # auto-remove colon for empty `desc`
                if not prefix:
                    bar_format = bar_format.replace("{desc}: ", '')

                # Interpolate supplied bar format with the dict
                if '{bar}' in bar_format:
                    # Format left/right sides of the bar, and format the bar
                    # later in the remaining space (avoid breaking display)
                    l_bar_user, r_bar_user = bar_format.split('{bar}')
                    l_bar = l_bar_user.format(**format_dict)
                    r_bar = r_bar_user.format(**format_dict)
                else:
                    # Else no progress bar, we can just format and return
                    return bar_format.format(**format_dict)

            # Formatting progress bar space available for bar's display
            if ncols:
                N_BARS = max(1, ncols - len(RE_ANSI.sub('', l_bar + r_bar)))
            else:
                N_BARS = 10

            # format bar depending on availability of unicode/ascii chars
            ascii = UTF_FMT
            nsyms = len(ascii) - 1
            bar_length, frac_bar_length = divmod(
                int(frac * N_BARS * nsyms), nsyms)

            bar = ascii[-1] * bar_length
            frac_bar = ascii[frac_bar_length]

            # whitespace padding
            if bar_length < N_BARS:
                full_bar = bar + frac_bar + \
                    ascii[0] * (N_BARS - bar_length - 1)
            else:
                full_bar = bar + \
                    ascii[0] * (N_BARS - bar_length)

            # Piece together the bar parts
            return l_bar + full_bar + r_bar

        elif bar_format:
            # user-specified bar_format but no total
            return bar_format.format(bar='?', **format_dict)
        else:
            # no total: no progressbar, ETA, just progress stats
            return ((prefix + ": ") if prefix else '') + \
                '{0}{1} [{2}, {3}{4}]'.format(
                    n_fmt, unit, elapsed_str, rate_fmt, postfix)

    @staticmethod
    def format_sizeof(num, suffix='', divisor=1000):
        """
        Formats a number (greater than unity) with SI Order of Magnitude
        prefixes.

        Parameters
        ----------
        num  : float
            Number ( >= 1) to format.
        suffix  : str, optional
            Post-postfix [default: ''].
        divisor  : float, optionl
            Divisor between prefixes [default: 1000].

        Returns
        -------
        out  : str
            Number with Order of Magnitude SI unit postfix.
        """
        for unit in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 999.5:
                if abs(num) < 99.95:
                    if abs(num) < 9.995:
                        return '{0:1.2f}'.format(num) + unit + suffix
                    return '{0:2.1f}'.format(num) + unit + suffix
                return '{0:3.0f}'.format(num) + unit + suffix
            num /= divisor
        return '{0:3.1f}Y'.format(num) + suffix

    @staticmethod
    def format_interval(t):
        """
        Formats a number of seconds as a clock time, [H:]MM:SS

        Parameters
        ----------
        t  : int
            Number of seconds.

        Returns
        -------
        out  : str
            [H:]MM:SS
        """
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        if h:
            return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
        else:
            return '{0:02d}:{1:02d}'.format(m, s)

    @staticmethod
    def ema(x, mu=None, alpha=0.3):
        """
        Exponential moving average: smoothing to give progressively lower
        weights to older values.

        Parameters
        ----------
        x  : float
            New value to include in EMA.
        mu  : float, optional
            Previous EMA value.
        alpha  : float, optional
            Smoothing factor in range [0, 1], [default: 0.3].
            Increase to give more weight to recent values.
            Ranges from 0 (yields mu) to 1 (yields x).
        """
        return x if mu is None else (alpha * x) + (1 - alpha) * mu

    @staticmethod
    def status_printer(file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover

        def fp_write(s):
            fp.write(_unicode(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            len_s = len(s)
            fp_write('\r' + s + (' ' * max(last_len[0] - len_s, 0)))
            last_len[0] = len_s

        return print_status

    @property
    def format_dict(self):
        """Public API for read-only member access"""
        return dict(
            n=self.n, total=self.total,
            elapsed=self._time() - self.start_t
            if hasattr(self, 'start_t') else 0,
            ncols=self.dynamic_ncols(self.fp)
            if self.dynamic_ncols else self.ncols,
            prefix=self.desc, ascii=self.ascii, unit=self.unit,
            unit_scale=self.unit_scale,
            rate=1 / self.avg_time if self.avg_time else None,
            bar_format=self.bar_format, postfix=self.postfix,
            unit_divisor=self.unit_divisor)


def trange(*args, **kwargs):
    """
    A shortcut for tqdm(xrange(*args), **kwargs).
    On Python3+ range is used instead of xrange.
    """
    return tqdm(_range(*args), **kwargs)



if __name__ == '__main__':

    for i in tqdm(iterable=range(1000), desc="epoch"):
        time.sleep(0.001)

    """
    sys.stderr输出
    """

