import requests
import re
from lxml import etree


url = "https://zhuanlan.zhihu.com/p/39972832"

def get_html(url):
    #设置请求头，否则无法获取网页内容
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.51"}
    r = requests.get(url, headers=headers)
    r.encoding = "utf-8"
    return r.text

def parse_items(raw_html):
    #使用xpath解析网页内容
    html = etree.HTML(raw_html)
    # 需要注意这里定位到的元素是一个包含字符串的列表，所以下面需要取第一个元素（'[0]'）
    result1 = [html.xpath('//*[@id="root"]/div/main/div/article/div[1]/div/ol[1]/li[%s]/img//@alt'
                          %(i))[0] for i in range(1,35)]
    result2 = [html.xpath('//*[@id="root"]/div/main/div/article/div[1]/div/ol[2]/li[%s]/img//@alt'
                          %(j))[0] for j in range(1,101)]
    # 将两个列表连起来
    result = result1+result2
    # 使用re库进行解析，具体结果类似Xpath结果
    # regex = re.compile(r'<li><img.*?alt="(.*?)".*?</li>')
    # result = re.findall(regex,raw_html)
    return result

if __name__ == '__main__':
    html = get_html(url)
    # print(html)
    items = parse_items(html)
    # print(items)
    # 用于计数
    num = 0
    # 存文件
    with open("formula.md", 'w') as f:
        for i,item in zip(range(1,135),items):
            print(item)
            # 使输出符合markdown的语法，便于直接显示公式
            f.write("%d. $$\n    "%(i)+item+"\n    $$\n\n")
            num+=1
    print("已写入%d条公式"%(num))

