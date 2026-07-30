"""Number and date rendering kernels exposed through a single C ABI unit."""

from std.sys.info import simd_width_of

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def copy_bytes(src: BPtr, start: Int, size: Int, dst: BPtr, pos: Int) -> Int:
    for i in range(size):
        dst[pos + i] = src[start + i]
    return pos + size


def digits10(value: Int64) -> Int:
    var n = Int64(value)
    var count = 1
    while n >= 10:
        n //= 10
        count += 1
    return count


def pow10(n: Int) -> Int64:
    var value = Int64(1)
    for _ in range(n):
        value *= 10
    return value


def append_uint(
    value: Int64, min_width: Int, dst: BPtr, pos: Int, limit: Int
) -> Int:
    var width = digits10(value)
    if width < min_width:
        width = min_width
    if pos + width > limit:
        return -1
    var divisor = pow10(width - 1)
    var remaining = value
    for i in range(width):
        var digit = remaining // divisor if divisor > 0 else remaining
        dst[pos + i] = UInt8(48 + Int(digit))
        remaining -= digit * divisor
        if divisor > 1:
            divisor //= 10
    return pos + width


def format_scaled_row(
    row: Int,
    values: IPtr,
    decimals: Int,
    min_frac: Int,
    min_int: Int,
    primary_group: Int,
    secondary_group: Int,
    use_group: Int,
    group: BPtr,
    group_len: Int,
    decimal: BPtr,
    decimal_len: Int,
    dst: BPtr,
    stride: Int,
    lengths: IPtr,
) -> Int:
    var integer = values[row]
    var fraction = lengths[row] if decimals > 0 else Int64(0)
    var width = digits10(integer)
    if width < min_int:
        width = min_int
    var frac_width = decimals
    while frac_width > min_frac and fraction % 10 == 0:
        fraction //= 10
        frac_width -= 1

    var pos = row * stride
    var limit = pos + stride
    var divisor = pow10(width - 1)
    var remaining = integer
    for i in range(width):
        if pos >= limit:
            return -1
        var digit = remaining // divisor if divisor > 0 else remaining
        dst[pos] = UInt8(48 + Int(digit))
        pos += 1
        remaining -= digit * divisor
        if divisor > 1:
            divisor //= 10
        var right = width - i - 1
        if (
            use_group != 0
            and right > 0
            and (
                right == primary_group
                or (
                    right > primary_group
                    and secondary_group > 0
                    and (right - primary_group) % secondary_group == 0
                )
            )
        ):
            if pos + group_len > limit:
                return -1
            pos = copy_bytes(group, 0, group_len, dst, pos)

    if frac_width > 0:
        if pos + decimal_len + frac_width > limit:
            return -1
        pos = copy_bytes(decimal, 0, decimal_len, dst, pos)
        var frac_divisor = pow10(frac_width - 1)
        for _ in range(frac_width):
            var digit = fraction // frac_divisor if frac_divisor > 0 else fraction
            dst[pos] = UInt8(48 + Int(digit))
            pos += 1
            fraction -= digit * frac_divisor
            if frac_divisor > 1:
                frac_divisor //= 10
    lengths[row] = Int64(pos - row * stride)
    return 0


@export("mb_format_scaled")
def mb_format_scaled(
    values_addr: Int,
    count: Int,
    decimals: Int,
    min_frac: Int,
    min_int: Int,
    primary_group: Int,
    secondary_group: Int,
    use_group: Int,
    group_addr: Int,
    group_len: Int,
    decimal_addr: Int,
    decimal_len: Int,
    dst_addr: Int,
    stride: Int,
    lengths_addr: Int,
) abi("C") -> Int:
    if count < 0 or decimals < 0 or decimals > 18 or min_frac < 0 or min_frac > decimals:
        return -2
    if min_int < 0 or primary_group < 0 or secondary_group < 0:
        return -2
    if group_len < 0 or decimal_len < 0 or stride <= 0:
        return -2
    if count == 0:
        return 0
    if (
        values_addr == 0
        or dst_addr == 0
        or lengths_addr == 0
        or (group_len > 0 and group_addr == 0)
        or (decimal_len > 0 and decimal_addr == 0)
    ):
        return -2
    var values = IPtr(unsafe_from_address=values_addr)
    var group = BPtr(unsafe_from_address=group_addr)
    var decimal = BPtr(unsafe_from_address=decimal_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var lengths = IPtr(unsafe_from_address=lengths_addr)
    var factor = pow10(decimals)

    for i in range(count):
        if values[i] < 0:
            return -2

    if decimals > 0:
        comptime W = simd_width_of[DType.float64]()
        var factor_vec = SIMD[DType.int64, W](factor)
        var i = 0
        while i + W <= count:
            var scaled = values.load[width=W](i)
            values.store(i, scaled // factor_vec)
            lengths.store(i, scaled % factor_vec)
            i += W
        while i < count:
            var scaled = values[i]
            values[i] = scaled // factor
            lengths[i] = scaled % factor
            i += 1

    for row in range(count):
        if format_scaled_row(
            row,
            values,
            decimals,
            min_frac,
            min_int,
            primary_group,
            secondary_group,
            use_group,
            group,
            group_len,
            decimal,
            decimal_len,
            dst,
            stride,
            lengths,
        ) != 0:
            return -1
    return 0


def append_name(
    slot: Int, data: BPtr, offsets: IPtr, dst: BPtr, pos: Int, limit: Int
) -> Int:
    var start = Int(offsets[slot * 2])
    var end = Int(offsets[slot * 2 + 1])
    if pos + end - start > limit:
        return -1
    return copy_bytes(data, start, end - start, dst, pos)


def append_fraction(micros: Int64, width: Int, dst: BPtr, pos: Int, limit: Int) -> Int:
    if pos + width > limit:
        return -1
    var cursor = pos
    if width <= 6:
        var divisor = pow10(6 - width)
        var rounded = micros // divisor
        var remainder = micros % divisor
        var half = divisor // 2
        if remainder > half or (remainder == half and rounded % 2 == 1):
            rounded += 1
        return append_uint(rounded, width, dst, cursor, limit)
    var scaled = micros * pow10(width - 6)
    return append_uint(scaled, width, dst, cursor, limit)


@export("mb_format_dates")
def mb_format_dates(
    fields_addr: Int,
    count: Int,
    ops_addr: Int,
    op_count: Int,
    names_addr: Int,
    names_len: Int,
    offsets_addr: Int,
    offsets_count: Int,
    literal_addr: Int,
    literal_len: Int,
    dst_addr: Int,
    stride: Int,
    lengths_addr: Int,
) abi("C") -> Int:
    if (
        count < 0
        or op_count < 0
        or names_len < 0
        or offsets_count < 0
        or literal_len < 0
        or stride <= 0
    ):
        return -2
    if count == 0:
        return 0
    if (
        fields_addr == 0
        or ops_addr == 0
        or dst_addr == 0
        or lengths_addr == 0
        or (names_len > 0 and names_addr == 0)
        or (offsets_count > 0 and offsets_addr == 0)
        or (literal_len > 0 and literal_addr == 0)
    ):
        return -2
    var fields = IPtr(unsafe_from_address=fields_addr)
    var ops = IPtr(unsafe_from_address=ops_addr)
    var names = BPtr(unsafe_from_address=names_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var literals = BPtr(unsafe_from_address=literal_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var lengths = IPtr(unsafe_from_address=lengths_addr)

    for oi in range(op_count):
        var op = Int(ops[oi * 3])
        var width = Int(ops[oi * 3 + 1])
        var arg = Int(ops[oi * 3 + 2])
        if op < 0 or op > 14 or width < 0 or arg < 0:
            return -2
        if op == 0 and (arg > literal_len or width > literal_len - arg):
            return -2
        if op == 3 or op == 5 or op == 13 or op == 14:
            var slots = 12 if op == 3 else (7 if op == 5 else 2)
            if arg > offsets_count or slots > offsets_count - arg:
                return -2
            for slot in range(arg, arg + slots):
                var start = Int(offsets[slot * 2])
                var end = Int(offsets[slot * 2 + 1])
                if start < 0 or end < start or end > names_len:
                    return -2

    for row in range(count):
        var base = row * 8
        var year = fields[base]
        var month = fields[base + 1]
        var day = fields[base + 2]
        var weekday = fields[base + 3]
        var hour = fields[base + 4]
        var minute = fields[base + 5]
        var second = fields[base + 6]
        var micros = fields[base + 7]
        if (
            month < 1
            or month > 12
            or day < 1
            or day > 31
            or weekday < 0
            or weekday > 6
            or hour < 0
            or hour > 23
            or minute < 0
            or minute > 59
            or second < 0
            or second > 59
            or micros < 0
            or micros > 999999
        ):
            return -2
        var pos = row * stride
        var limit = pos + stride

        for oi in range(op_count):
            var op = Int(ops[oi * 3])
            var width = Int(ops[oi * 3 + 1])
            var arg = Int(ops[oi * 3 + 2])
            if op == 0:
                if pos + width > limit:
                    return -1
                pos = copy_bytes(literals, arg, width, dst, pos)
            elif op == 1:
                var y = year % 100 if width == 2 else year
                var pad = 2 if width == 2 else width
                pos = append_uint(y, pad, dst, pos, limit)
            elif op == 2:
                pos = append_uint(month, width, dst, pos, limit)
            elif op == 3:
                pos = append_name(arg + Int(month) - 1, names, offsets, dst, pos, limit)
            elif op == 4:
                pos = append_uint(day, width, dst, pos, limit)
            elif op == 5:
                pos = append_name(arg + Int(weekday), names, offsets, dst, pos, limit)
            elif op == 6:
                pos = append_uint(hour, width, dst, pos, limit)
            elif op == 7:
                var h = hour % 12
                if h == 0:
                    h = 12
                pos = append_uint(h, width, dst, pos, limit)
            elif op == 8:
                pos = append_uint(hour % 12, width, dst, pos, limit)
            elif op == 9:
                var h = hour
                if h == 0:
                    h = 24
                pos = append_uint(h, width, dst, pos, limit)
            elif op == 10:
                pos = append_uint(minute, width, dst, pos, limit)
            elif op == 11:
                pos = append_uint(second, width, dst, pos, limit)
            elif op == 12:
                pos = append_fraction(micros, width, dst, pos, limit)
            elif op == 13:
                pos = append_name(arg + (1 if hour >= 12 else 0), names, offsets, dst, pos, limit)
            elif op == 14:
                pos = append_name(arg + (1 if year >= 0 else 0), names, offsets, dst, pos, limit)
            if pos < 0:
                return -1
        lengths[row] = Int64(pos - row * stride)
    return 0


def civil_from_days(days: Int64) -> Tuple[Int64, Int64, Int64]:
    var z = days + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var year = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var day = doy - (153 * mp + 2) // 5 + 1
    var month = mp + (Int64(3) if mp < 10 else Int64(-9))
    year += Int64(1) if month <= 2 else Int64(0)
    return (year, month, day)


@export("mb_decompose_days")
def mb_decompose_days(days_addr: Int, count: Int, fields_addr: Int) abi("C") -> Int:
    if count < 0:
        return -2
    if count == 0:
        return 0
    if days_addr == 0 or fields_addr == 0:
        return -2
    var days = IPtr(unsafe_from_address=days_addr)
    var fields = IPtr(unsafe_from_address=fields_addr)
    for i in range(count):
        var y, m, d = civil_from_days(days[i])
        var weekday = (days[i] + 3) % 7
        if weekday < 0:
            weekday += 7
        fields[i * 8] = y
        fields[i * 8 + 1] = m
        fields[i * 8 + 2] = d
        fields[i * 8 + 3] = weekday
        fields[i * 8 + 4] = 0
        fields[i * 8 + 5] = 0
        fields[i * 8 + 6] = 0
        fields[i * 8 + 7] = 0
    return 0
