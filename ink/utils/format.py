def get_entity(path, tag_map):
    results = []
    record = {}
    for index, tag_id in enumerate(path):

        if tag_id == 0:  # 0是我们的pad label
            continue

        tag = tag_map[tag_id]
        if tag.startswith("B_"):
            if record.get('end'):
                if (record['type'] != 'T'):
                    results.append(record)
            record = {}
            record['begin'] = index
            record['type'] = tag.split('_')[1]
        elif tag.startswith('I_') and 'begin' in record:
            tag_type = tag.split('_')[1]
            if tag_type == record['type']:
                record['end'] = index
        else:
            if record.get('end'):  # 把Time过滤掉
                if (record['type'] != 'T'):
                    results.append(record)
                record = {}
    if record.get('end'):
        if (record['type'] == 'T'):
            pass
        else:
            results.append(record)
    return results
