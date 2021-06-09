from .class_list import classlist


def extract_entities(labels_lst, start_label="1_4"):
    def gen_entities(label_lst, start_label="1", dims=1):
        entities = dict()

        if "_" in start_label:
            start_label = start_label.split("_")
            start_label = [int(tmp) for tmp in start_label]
            indicator = sum([int(tmp) for tmp in (bool(label in start_label) for label in label_lst)])
        else:
            start_label = int(start_label)
            indicator = 1 if start_label in labels_lst else 0

        if indicator > 0:
            if isinstance(start_label, list):
                ixs, _ = zip(*filter(lambda x: x[1] in start_label, enumerate(label_lst)))
            elif isinstance(start_label, int):
                ixs, _ = zip(*filter(lambda x: x[1] == start_label, enumerate(label_lst)))
            else:
                raise ValueError("You Should Notice that The FORMAT of your INPUT")

            ixs = list(ixs)
            ixs.append(len(label_lst))
            for i in range(len(ixs) - 1):
                end_ix = ixs[i + 1]
                entities["{}_{}".format(ixs[i], end_ix)] = label_lst[ixs[i]: end_ix]
        return entities

    entities = gen_entities(labels_lst, start_label=start_label)

    return entities


def split_index(label_list):
    label_dict = {label: i for i, label in enumerate(label_list)}
    label_idx = [tmp_value for tmp_key, tmp_value in label_dict.items() if "S" in tmp_key.split("-")[0] or "B" in tmp_key]
    str_label_idx = [str(tmp) for tmp in label_idx]
    label_idx = "_".join(str_label_idx)
    return label_idx


def format_output(pred_label, label_list):
    start_label = split_index(label_list)
    pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label)][1:-1]
    pred_class = [classlist[i] for i in pred_label]
    entities = extract_entities(pred_label, start_label=start_label)
    entities_range = [tuple(map(int, key.split('_'))) for key in entities.keys()]
    return [pred_class, [(r, pred_class[r[0]][2:] if pred_class[r[0]] != 'O' else 'O') for r in entities_range]]