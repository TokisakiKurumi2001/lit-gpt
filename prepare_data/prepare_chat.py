from __future__ import annotations
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChatNode:
    text: str
    parent: Optional[ChatNode] = None

storage = []

def process_node_data(data, parent_node: Optional[ChatNode] = None):
    # data is the JSON objects that has the following keys
    # 'message_id', 'parent_id', 'user_id', 'created_date', 'text', 'role', 'lang', 'review_count', 'review_result', 'deleted', 'rank', 'synthetic', 'emojis', 'replies', 'labels', 'detoxify'
    text = data['text']
    chatnode = ChatNode(text, parent_node)
    # breakpoint()

    # stopping condition
    if len(data['replies']) == 0:
        storage.append(chatnode)
        return None

    for node in data['replies']:
        process_node_data(node, chatnode)

if __name__ == "__main__":
    with open('2023-04-12_oasst_ready.trees.jsonl') as fin:
        for line in tqdm(fin):
            data = json.loads(line)
            data = data['prompt']
            if data['lang'] != 'en':
                continue
            process_node_data(data)
    
    res = []
    for el in storage:
        stack = []
        traverse = el
        while traverse is not None:
            stack.append(traverse.text)
            traverse = traverse.parent

        res.append({"chat": stack[::-1]})

    with open('chat_data.jsonl', 'w') as fout:
        for chat in res:
            fout.write(json.dumps(chat)+'\n')

