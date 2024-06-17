def parse_freeze_strategy(strategy):
    action, layers = strategy.split(':', maxsplit=1)
    if layers.startswith('[') and layers.endswith(']'):
        layer_list = []
        for i in layers.strip('][').split(','):
            if '-' in i:
                start, end = i.split('-')
                start, end = int(start), int(end)
                layer_list.extend(list(range(start, end+1)))
            else:
                layer_list.append(int(i))
    elif len(layers.split('-')) == 3:
        layer_list = []
        start, end, step = [int(i) for i in layers.split('-')]
        layer_list = list(range(start, end+1, step))
    else:
        raise Exception('冻结策略格式有误')
    return action, layer_list

if __name__ == '__main__':
    print(parse_freeze_strategy('active:9-90-9'))