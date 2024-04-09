import sys

DELIMITER = '.'
DELIMITER_NUM = 13
SPLITSYMBOL = '|'
SHIFTER_NUM = 198
COLON_NUM = 25
END_SYMBOL = '|25|.'

# 当前仅支持qwen词表id，这里列举了部分和\n合并的tokenid，待补充
#     ：\n   28311
#     ：\n\n 48443
#     ？\n   94432
#     \n\n   271
#     ，\n   41453
#     。\n   8997
#     .\n    624
#     ,\n    345
#     \n\n\n 1406
#     !\n    4894
#      \n    715
#     \n     198

class Node:
    def __init__(self):
        self.children = {}
        self.direct_end_list = []
    def get_child(self, key):
        if key in self.children:
            return self.children[key]
        else:
            return None

    def add_child(self, key, child):
        self.children[key] = child

    def set_direct_end_list(self, input):
        temp = input[:-2]
        self.direct_end_list.append(temp)

    def get_direct_end_list(self):
        return self.direct_end_list

class ResourceTree:
    def __init__(self):
        self.root = Node()
    def ends_with(self, string, suffix):
        if len(string) < len(suffix):
            return False
        return string[-len(suffix):] == suffix

    def add_resource(self, resource):
        curr = self.root
        delimiter = DELIMITER
        if not self.ends_with(resource, END_SYMBOL):
            curr.set_direct_end_list(resource)
        while delimiter in resource:
            token, resource = resource.split(delimiter, 1)
            if curr.get_child(token) is None:
                curr.add_child(token, Node())
            curr = curr.get_child(token)

    def remove_symbol(self, string, symbol):
        result = string
        if result and result[0] == symbol:
            result = result[1:]
        if result and result[-1] == symbol:
            result = result[:-1]
        return result

    def find_resource(self, input):
        curr = self.root
        delimiter = DELIMITER
        father_key = ""
        direct_end_list = curr.get_direct_end_list()
        match_res = []
        for item in direct_end_list:
            if input == item:
                return match_res

        while delimiter in input:
            token, input = input.split(delimiter, 1)
            father_key += token + DELIMITER
            curr = curr.get_child(token)
            if curr is None:
                print("Resource not found")
                return match_res

        for key, value in curr.children.items():
            temp = self.remove_symbol(key, SPLITSYMBOL)
            input = self.remove_symbol(input, SPLITSYMBOL)
            if temp.startswith(input):
                if temp == input:
                    match_res.append(str(13))
                else:
                    father_key = self.remove_symbol(father_key, SPLITSYMBOL)
                    temp = father_key + SPLITSYMBOL + temp
                    match_res.append(temp)

        return match_res

class RestrictDecode:

    def __init__(self):
        self.resourceTree = ResourceTree()
    def init(self, res_path):
        try:
            with open(res_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    self.resourceTree.add_resource(line.strip())
            print("load res success")
            return 0
        except IOError:
            print("Unable to open res file:", res_path)
            return -1

    def remove_symbol(self, string, symbol):
        result = string
        if result and result[0] == symbol:
            result = result[1:]
        if result and result[-1] == symbol:
            result = result[:-1]
        return result

    def subtract_strings(self, a, b):
        found = a.find(b)
        if found != -1:
            a = a[:found] + a[found + len(b):]
        return a

    def split(self, input):
        ret_vector = []
        for item in input.split(SPLITSYMBOL):
            ret_vector.append(int(item))
        return ret_vector

    def remove_duplicates(self, input):
        unique_elements = set(input)
        return list(unique_elements)

    def get_last_substring(self, string, delimiter):
        pos = string.rfind(delimiter)
        if pos != -1:
            return string[pos + 1:]
        if string == "":
            return "0"
        return string

    def get_next_tokenid(self, cur_tokenid):
        ret = []
        try:
            cur_last_tokenid = int(self.get_last_substring(cur_tokenid, SPLITSYMBOL))
        except:
            cur_last_tokenid = 0
        next_tokenids = self.resourceTree.find_resource(cur_tokenid)
        if next_tokenids:
            for item in next_tokenids:
                next_tokenids_temp = self.remove_symbol(item, SPLITSYMBOL)
                result = self.subtract_strings(next_tokenids_temp, cur_tokenid)
                result = self.remove_symbol(result, SPLITSYMBOL)
                tokenids = self.split(result)
                if tokenids[0] == cur_last_tokenid:
                    ret.append(tokenids[1])
                else:
                    ret.append(tokenids[0])
        ret = self.remove_duplicates(ret)
        return ret

    def split_array(self, arr, flag):
        result = []
        temp = []
        for item in arr:
            if item != flag:
                temp.append(item)
            else:
                result.append(temp)
                temp = []
        result.append(temp)
        return result

    def split_array_by_vec(self, nums, special):
        result = []
        temp = []

        for item in nums:
            if item in special:
                if temp:
                    result.append(temp)
                    temp = []
            else:
                temp.append(item)

        if temp:
            result.append(temp)

        if nums[-1] in special:
            result.append([])

        return result

    def process(self, whole_summary):
        tokenids = []

        if not whole_summary:
            return tokenids

        SHIFTER_VEC = [28311, 48443, 94432, 271, 41453, 8997, 624, 345, 1406, 4894, 715, 198]
        frag_summarys = self.split_array_by_vec(whole_summary, SHIFTER_VEC)

        keys = []
        if frag_summarys[-1]:
            keys = self.split_array(frag_summarys[-1], COLON_NUM)
            if len(keys) > 1:
                return tokenids

        inpit_str = ""
        if not keys:
            tokenids = self.get_next_tokenid(inpit_str)
            return tokenids

        if not keys[0]:
            return tokenids

        for item in keys[0]:
            convert = DELIMITER if item == DELIMITER_NUM else str(item)
            inpit_str += SPLITSYMBOL + convert

        inpit_str = self.remove_symbol(inpit_str, SPLITSYMBOL)
        tokenids = self.get_next_tokenid(inpit_str)
        return tokenids

if __name__ == '__main__':
    restrictdecode = RestrictDecode()
    restrictdecode.init("./resource_out.txt")
    whole_summary = [110928, 45785, 25, 17, 15, 17, 18, 7948, 23, 9754, 17, 23, 198, 100022, 55338, 106073, 1447, 103998, 25, 220, 104935, 52801, 101437, 101036, 100147, 100540, 715, 101924, 25, 220, 104935, 52801, 101437, 101036, 49567, 100158, 35946, 102115, 46944, 42411, 102115, 104882, 220, 56568, 101095, 715, 106073, 25, 67949, 105051, 15946, 101068, 101051, 60610, 33071, 27369, 198, 103998, 25, 220, 56568, 99487, 99190, 101056, 20412, 99459, 110237, 104256, 715, 101924, 25, 49434, 239, 101056, 80158, 104037, 104037, 99180, 103188, 104037, 112173, 102706, 9370, 102838, 10236, 226, 114, 33447, 104854, 99405, 27442, 99471, 80158, 107554, 32181, 247, 18947, 20450, 45861, 27442, 100003, 42411, 17447, 100634, 34187, 32181, 247, 97084, 71268, 104169, 105603, 99817, 99319, 34187, 32181, 247, 69249, 104044, 101114, 715, 106073, 25, 46451, 99252, 99497, 13, 20450, 16, 13, 99558, 101368, 13, 101368, 116925, 25, 100277, 102706, 198, 105185, 105262, 13]
    tokenids = restrictdecode.process(whole_summary)
    print(tokenids)
