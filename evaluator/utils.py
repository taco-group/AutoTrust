def list2dict(lst, open_ended=False):
    return {item['question_id']: item for item in lst if open_ended or item['question_type'] != 'open-ended'}