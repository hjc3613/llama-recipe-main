from loguru import logger

# first 是关于 UTF-8 字符中首字节的编码信息。
# 将所有的首字节进行分类，分为：-1、as、1、2、3、4、5、6、7 九类，
# 其中 -1 代表无效首字节，1 代表双字节字符的首字节
# 2、3、4 代表三字节字符的首字节
# 5、6、7 代表四字节字符的首字节
first = [
    #   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0x00-0x0F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0x10-0x1F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0x20-0x2F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0x30-0x3F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0x40-0x4F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0x50-0x5F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0x60-0x6F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 0x70-0x7F
    #   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 0x80-0x8F
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 0x90-0x9F
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 0xA0-0xAF
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 0xB0-0xBF
    -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 0xC0-0xCF
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 0xD0-0xDF
    2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3,  # 0xE0-0xEF
    5, 6, 6, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 0xF0-0xFF
]


def is_valid_symbol(byte_arr):
    flag = first[byte_arr[0]]
    if flag < 0:
        return False

    arr_len = len(byte_arr)
    if flag == 0 and arr_len == 1:
        return True
    if flag == 1 and arr_len == 2:
        if first[byte_arr[1]] == -1:
            return True
    if flag >= 2 and flag <= 4 and arr_len == 3:
        if byte_arr == bytes.fromhex('efbfbd'):  # replacement character UTF16: U+FFFD
            return False
        if first[byte_arr[1]] == -1 and first[byte_arr[2]] == -1:
            return True
    if flag >= 5 and flag <= 7 and arr_len == 4:
        if first[byte_arr[1]] == -1 and first[byte_arr[2]] == -1 and first[byte_arr[3]] == -1:
            return True

    return False


def is_leading_byte2(char):
    a = first[char]
    if a >= 0:
        return True
    else:
        return False


def get_complete_str_simple(input_str):
    byte_str = input_str.encode("utf-8")
    # byte_str = b'\xe9\xa6\x88\xef\xbf\xbd'
    pattern = b'\xef\xbf\xbd'
    pattern_len = len(pattern)
    # print('***',input_str)
    # print(byte_str)

    index = byte_str.find(pattern)
    while index == 0:  # only the first pattern
        byte_str = byte_str[index + len(pattern):]
        index = byte_str.find(pattern)

    index = byte_str.rfind(pattern)
    while index > 0:
        byte_str = byte_str[:index]
        index = byte_str.rfind(pattern)

    # print(byte_str)
    return byte_str.decode("utf-8")


def get_complete_str(input_str):
    byte_str = input_str.encode("utf-8")

    # byte_str = b'\xe9\xa6\x88\xef\xbf\xbd\xef\xbf\xbd'
    # nput_str = byte_str.decode("utf-8")
    # print('***',input_str)
    # print(byte_str)
    str_len = len(byte_str)
    end_pos = str_len
    for i in range(str_len - 1, -1, -1):
        if is_leading_byte2(byte_str[i]):
            flag = is_valid_symbol(byte_str[i:end_pos])
            if flag:
                return byte_str[:end_pos].decode("utf-8")
            else:
                end_pos = i
    return ''


# a='111'
# print('*****',get_complete_str_simple(a))


# https:#www.cnblogs.com/golove/p/5889790.html
def is_leading_byte(char):
    # 位标记（用于判断字节有效性）
    t1 = 0x00  # 0000 0000 单字节字符的首字节标记（二进制以 0     开头）
    mask1 = 0x80  # 1000 0000 所有字符的后续字节标记（二进制以 10    开头）
    t2 = 0xC0  # 1100 0000 双字节字符的首字节标记（二进制以 110   开头）
    mask2 = 0xE0  # mask 1110 0000
    t3 = 0xE0  # 1110 0000 三字节字符的首字节标记（二进制以 1110  开头）
    mask3 = 0xF0  # 1111 0000
    t4 = 0xF0  # 1111 0000 四字节字符的首字节标记（二进制以 11110 开头）
    mask4 = 0xF8  # 1111 1000
    # t5 = 0xF8 # 1111 1000 好像未使用
    f1 = ((mask1 & char) == t1)
    print(f1)
    f2 = ((mask2 & char) == t2)
    print(f2)
    f3 = ((mask3 & char) == t3)
    print(f3)
    f4 = ((mask4 & char) == t4)
    print(f4)


def get_inc_substr(basestr, newstr):
    if basestr == newstr:
        return ''
    index = newstr.find(basestr)
    if index == -1:
        logger.debug('error')
        return newstr
    return newstr[index + len(basestr):]

# get_new_substr('aaaa','aaa')

# for char in b:
# is_leading_byte(char)
#    print(is_leading_byte2(char))

