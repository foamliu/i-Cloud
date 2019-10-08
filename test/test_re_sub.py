import re

if __name__ == '__main__':
    s = "标点符号，书面上用于标明句读和语气的符号。“标点符号是辅助文字记录语言的符号，是书面语的"
    replaced = re.sub('[。？！，、；：]', ' ', s)
    replaced = re.sub('[“”（）《》〈〉]', '', replaced)
    print(replaced)
