import json

def get_required_batches(df):
    required_batches = []
    counter = len(df)
    i = 1
    while counter >= 25:
        required_batches.append(i)
        i = i + 1
        counter = counter - 25

    if counter == 0:
        required_batches.append(i)

    return required_batches


def write_log(completed_batches, batch_size, classification_log_path):

    data = {"completed_batches": completed_batches, "batch_size": batch_size}
    jstr = json.dumps(data, indent=4)
    with open(classification_log_path, 'w', encoding='utf-8') as outfile:
        json.dump(jstr, outfile, ensure_ascii=False)