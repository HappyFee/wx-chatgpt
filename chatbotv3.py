"""
A simple wrapper for the official ChatGPT API
"""
import argparse
import json
import os
import sys
import urllib

import requests
import tiktoken

from utils import create_completer
from utils import create_session
from utils import get_input

ENGINE = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo"
ENCODER = tiktoken.get_encoding("gpt2")


class Chatbot:
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str,
        engine: str = None,
        proxy: str = None,
        max_tokens: int = 3000,
        temperature: float = 0.5,
        top_p: float = 1.0,
        reply_count: int = 1,
        system_prompt: str = "你是全能助手小濛濛，善于解决一切问题",
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        self.engine = engine or ENGINE
        self.session = requests.Session()
        self.api_key = api_key
        self.proxy = proxy
        if self.proxy:
            proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            self.session.proxies = proxies
        self.conversation: dict = {
            "default": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
        }
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reply_count = reply_count

        initial_conversation = "\n".join(
            [x["content"] for x in self.conversation["default"]],
        )
        if len(ENCODER.encode(initial_conversation)) > self.max_tokens:
            raise Exception("System prompt is too long")

    def add_to_conversation(self, message: str, role: str, convo_id: str = "default"):
        """
        Add a message to the conversation
        """
        self.conversation[convo_id].append({"role": role, "content": message})

    def __truncate_conversation(self, convo_id: str = "default"):
        """
        Truncate the conversation
        """
        while True:
            full_conversation = "\n".join(
                [x["content"] for x in self.conversation[convo_id]],
            )
            if (
                len(ENCODER.encode(full_conversation)) > self.max_tokens
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                self.conversation[convo_id].pop(1)
            else:
                break

    def ask_stream(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        **kwargs,
    ) -> str:
        """
        Ask a question
        """
        self.add_to_conversation(prompt, "user", convo_id=convo_id)
        self.__truncate_conversation(convo_id=convo_id)
        # Get response
        response = self.session.post(
            #"https://api.openai.com/v1/chat/completions",
            "https://service-9k8pqprg-1302498424.jp.apigw.tencentcs.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
            json={
                "model": self.engine,
                "messages": self.conversation[convo_id],
                "stream": True,
                # kwargs
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "n": kwargs.get("n", self.reply_count),
                "user": role,
            },
            stream=True,
        )
        if response.status_code != 200:
            raise Exception(
                f"Error: {response.status_code} {response.reason} {response.text}",
            )
        response_role: str = None
        full_response: str = ""
        for line in response.iter_lines():
            if not line:
                continue
            # Remove "data: "
            line = line.decode("utf-8")[6:]
            if line == "[DONE]":
                break
            resp: dict = json.loads(line)
            choices = resp.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta")
            if not delta:
                continue
            if "role" in delta:
                response_role = delta["role"]
            if "content" in delta:
                content = delta["content"]
                full_response += content
                yield content
        self.add_to_conversation(full_response, response_role, convo_id=convo_id)

    def ask(self, prompt: str, role: str = "user", convo_id: str = "default", **kwargs):
        """
        Non-streaming ask
        """
        response = self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            **kwargs,
        )
        full_response: str = "".join(response)
        return full_response

    def rollback(self, n: int = 1, convo_id: str = "default"):
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None):
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {"role": "system", "content": system_prompt or self.system_prompt},
        ]

    def save(self, file: str):
        """
        Save the conversation to a JSON file
        """
        try:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(self.conversation, f, indent=2)
        except FileNotFoundError:
            print(f"Error: {file} cannot be created")

    def load(self, file: str):
        """
        Load the conversation from a JSON  file
        """
        try:
            with open(file, encoding="utf-8") as f:
                self.conversation = json.load(f)
        except FileNotFoundError:
            print(f"Error: {file} does not exist")

    def print_config(self, convo_id: str = "default"):
        """
        Prints the current configuration
        """
        print(
            f"""
ChatGPT Configuration:
  Messages:         {len(self.conversation[convo_id])} / {self.max_tokens}
  Engine:           {self.engine}
  Temperature:      {self.temperature}
  Top_p:            {self.top_p}
  Reply count:      {self.reply_count}
            """,
        )

    def print_help(self):
        """
        Prints the help message
        """
        print(
            """
Commands:
  !help           Display this message
  !rollback n     Rollback the conversation by n messages
  !save filename  Save the conversation to a file
  !load filename  Load the conversation from a file
  !reset          Reset the conversation
  !exit           Quit chat
Config Commands:
  !config         Display the current config
  !temperature n  Set the temperature to n
  !top_p n        Set the top_p to n
  !reply_count n  Set the reply_count to n
  !engine engine  Sets the chat model to engine
  """,
        )

    def handle_commands(self, input: str, convo_id: str = "default") -> bool:
        """
        Handle chatbot commands
        """
        command, *value = input.split(" ")
        if command == "!help":
            self.print_help()
        elif command == "!exit":
            exit()
        elif command == "!reset":
            self.reset(convo_id=convo_id)
            print("\nConversation has been reset")
        elif command == "!config":
            self.print_config(convo_id=convo_id)
        elif command == "!rollback":
            self.rollback(int(value[0]), convo_id=convo_id)
            print(f"\nRolled back by {value[0]} messages")
        elif command == "!save":
            self.save(value[0])
            print(f"\nConversation has been saved to {value[0]}")
        elif command == "!load":
            self.load(value[0])
            print(
                f"\n{len(self.conversation[convo_id])} messages loaded from {value[0]}",
            )
        elif command == "!temperature":
            self.temperature = float(value[0])
            print(f"\nTemperature set to {value[0]}")
        elif command == "!top_p":
            self.top_p = float(value[0])
            print(f"\nTop_p set to {value[0]}")
        elif command == "!reply_count":
            self.reply_count = int(value[0])
            print(f"\nReply count set to {value[0]}")
        elif command == "!engine":
            self.engine = value[0]
            print(f"\nEngine set to {value[0]}")
        else:
            return False

        return True
    
    def get_data(self, page_list, page_data, err_count, err_list):

        show_template = ["bond_id", "bond_nm", "price", "increase_rt","stock_id","stock_nm","sprice","sincrease_rt","convert_price","convert_value","premium_rt","dblow","adjust_condition","rating_cd","force_redeem_price","convert_amt_ratio","short_maturity_dt","year_left","curr_iss_amt","ytm_rt","bond_nm_tip","convert_price_tips","convert_cd_tip","ref_yield_info"]
        for ele in page_list:
            tmp = list(filter(lambda x: x['bond_id']== ele['bond_id'], page_data))
            if len(tmp)> 0:
                tmp = tmp[0]
            else:
                #print("error", ele["bond_id"])#, ele)
                err_count += 1
                err_list.append(ele["bond_id"])
            for template in show_template:
                if template not in ele and template in tmp:
                    ele[template] = tmp[template]

        return err_count, err_list

    def jisilu(self) -> None:
        
        headers = {
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36',
            'Referer':"https://www.jisilu.cn/web/data/cb/list",
            #'Columns': '1,70,2,3,5,6,11,12,14,15,16,29,30,32,34,35,75,44,46,47,52,53,54,56,57,58,59,60,62,63,67',
            #'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            #'Cookie': 'Hm_lvt_164fe01b1433a19b507595a43bf58262=1627209458,1627209724,1627216989,1627915638; kbzw__Session=k86o56au3ckn0hvsohj2mmk8e0',
            }
        #usr_name, password, return_url= "https://www.jisilu.cn/", auto_login = 1, aes = 1
        #登录
        login_url = 'https://www.jisilu.cn/webapi/account/login_process/'
        session = requests.Session()
        data = {
            'ase':1,
            'auto_login': 1,
            'password': "0e0b00d7ca83c69ccb66b4a8ba426921",
            'return_url': "https://www.jisilu.cn/",
            'user_name': "6a33b283968dc30895abf2ef691aca35"
        }
        login_txt = session.post(url = login_url, data = data, headers = headers ).json()
        if login_txt["code"] == 200:
            pass
        else:
            print("error")

#        print('login_txt:', login_txt)

        url= 'https://www.jisilu.cn/webapi/cb/list_new/'
        url2 = 'https://www.jisilu.cn/webapi/cb_chart/industry_group/'
        page_list = session.get(url=url, headers=headers).json()['data']
        page_data = session.get(url=url2, headers=headers).json()['data']
        #print(list(filter(lambda x: x['bond_id'] == '123070', page_list['data'])))
        #print(len(page_data))
        err_count = 0
        err_list = []

        #print(page_list, page_data)
        all_template = {"bond_id":"代码","bond_nm":"名称","price":"现价","increase_rt":"涨跌幅","stock_id":"正股编码","stock_nm":"正股名称","stock_py":"正股缩写","sprice":"正股价格","sincrease_rt":"正股涨跌","pb":"正股PB","convert_price":"转股价","convert_value":"转股价值","premium_rt":"转股溢价率","dblow":"双低","adjust_condition":"下修条件","rating_cd":"评级","put_convert_price":"回售触发价","force_redeem_price":"强赎触发价","convert_amt_ratio":"转债流通市值占比","short_maturity_dt":"到期时间","year_left":"剩余年限","curr_iss_amt":"剩余规模（亿元）","volume":"成交额（万元）","turnover_rt":"换手率","ytm_rt":"到期税前收益率","province":"省份","bond_nm_tip":"转债提示","convert_price_tips":"转股价格提示","convert_cd_tip":"转股提示","ref_yield_info":"信息","adjusted":"已调整","price_tips":"价格提示"}
        err_count, err_list = self.get_data(page_list, page_data,err_count, err_list)
        save_data = []
        show_data = []
        for data in page_list:
            tmp = {}
            save_tmp = {}
            for id, mem in data.items():
                if id in all_template.keys():

                    tmp[all_template[id]] = mem
                    save_tmp[id] = mem
                else:
                    tmp[id] = mem
                    save_tmp[id] = mem
            show_data.append(tmp)
            save_data.append(save_tmp)
        count = 0
        no_name_num = 0
        no_name_list = []
        filter_nondustry_data = []
        for i, ele in enumerate(show_data):
            if "名称" in ele.keys():
                #print(ele["名称"])
                count += 1
                filter_nondustry_data.append(save_data[i])
            else:
                no_name_list.append(ele['代码'])
                no_name_num += 1

        print("No name num:", no_name_num)
        print("Have name number:", count)
        print("dustry data don't have number:", err_count)
        print("All number of dustry data:", len(page_data))
        print("The bond_id of  not have name:", no_name_list)
        #"dblow":"双低""year_left":"剩余年限","curr_iss_amt":"剩余规模（亿元）",

        order_data = sorted(filter_nondustry_data, key = lambda x:x["dblow"])
        filter_data = list(filter(lambda x:x["year_left"]>1 and x["curr_iss_amt"]<= 10 and x["price"] < 130, order_data))
        count_dblow = 0
        res = []
        for ele in filter_data[:15]:
            temp = []
            if ele["bond_nm"][2:] == "转债":
                temp.append(ele["bond_nm"][:2])
                temp.append(all_template["price"])
                temp.append(ele["price"])
                temp.append(all_template["premium_rt"])
                temp.append(ele["premium_rt"])
                
                res.append(temp)
                print(ele["bond_nm"][:2], end = ' ')
            else:
                temp.append(ele["bond_nm"][:2])
                temp.append(all_template["price"])
                temp.append(ele["price"])
                temp.append(all_template["premium_rt"])
                temp.append(ele["premium_rt"])
                
                res.append(temp)
                print(ele["bond_nm"], end = ' ')

            count_dblow += 1
        return res



def main():
    """
    Main function
    """
    print(
        """
    ChatGPT - Official ChatGPT API
    Repo: github.com/acheong08/ChatGPT
    """,
    )
    print("Type '!help' to show a full list of commands")
    print("Press Esc followed by Enter or Alt+Enter to send a message.\n")

    # Get API key from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for response",
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable streaming",
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default="你是全能助手小濛濛，善于解决一切问题",
        help="Base prompt for chatbot",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy address",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
        help="Top p for response",
    )
    parser.add_argument(
        "--reply_count",
        type=int,
        default=1,
        help="Number of replies for each prompt",
    )
    parser.add_argument(
        "--enable-internet",
        action="store_true",
        help="Allow ChatGPT to search the internet",
    )
    args = parser.parse_args()
    # Initialize chatbot
    chatbot = Chatbot(
        api_key=args.api_key,
        system_prompt=args.base_prompt,
        proxy=args.proxy,
        temperature=args.temperature,
        top_p=args.top_p,
        reply_count=args.reply_count,
    )
    # Check if internet is enabled
    if args.enable_internet:
        chatbot.system_prompt = """
        You are ChatGPT, an AI assistant that can access the internet. Internet search results will be sent from the system in JSON format.
        Respond conversationally and cite your sources via a URL at the end of your message.
        """
        chatbot.reset(
            convo_id="search",
            system_prompt='For given prompts, summarize it to fit the style of a search query to a search engine. If the prompt cannot be answered by an internet search, is a standalone statement, is a creative task, is directed at a person, or does not make sense, type "none". DO NOT TRY TO RESPOND CONVERSATIONALLY. DO NOT TALK ABOUT YOURSELF. IF THE PROMPT IS DIRECTED AT YOU, TYPE "none".',
        )
        chatbot.add_to_conversation(
            message="What is the capital of France?",
            role="user",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="Capital of France",
            role="assistant",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="Who are you?",
            role="user",
            convo_id="search",
        )
        chatbot.add_to_conversation(message="none", role="assistant", convo_id="search")
        chatbot.add_to_conversation(
            message="Write an essay about the history of the United States",
            role="user",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="none",
            role="assistant",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="What is the best way to cook a steak?",
            role="user",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="How to cook a steak",
            role="assistant",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="Hello world",
            role="user",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="none",
            role="assistant",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="Who is the current president of the United States?",
            role="user",
            convo_id="search",
        )
        chatbot.add_to_conversation(
            message="United States president",
            role="assistant",
            convo_id="search",
        )
    session = create_session()
    completer = create_completer(
        [
            "!help",
            "!exit",
            "!reset",
            "!rollback",
            "!config",
            "!engine",
            "!temperture",
            "!top_p",
            "!reply_count",
            "!save",
            "!load",
        ],
    )
    # Start chat
    while True:
        print()
        try:
            print("User: ")
            prompt = get_input(session=session, completer=completer)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit()
        if prompt.startswith("!") and chatbot.handle_commands(prompt):
            continue
        print()
        print("ChatGPT: ", flush=True)
        if args.enable_internet:
            query = chatbot.ask(
                f'This is a prompt from a user to a chatbot: "{prompt}". Respond with "none" if it is directed at the chatbot or cannot be answered by an internet search. Otherwise, respond with a possible search query to a search engine. Do not write any additional text. Make it as minimal as possible',
                convo_id="search",
                temperature=0.0,
            ).strip()
            print("Searching for: ", query, "")
            # Get search results
            if query == "none":
                search_results = '{"results": "No search results"}'
            else:
                search_results = requests.post(
                    url="https://ddg-api.herokuapp.com/search",
                    json={"query": query, "limit": 3},
                    timeout=10,
                ).text
            print(json.dumps(json.loads(search_results), indent=4))
            chatbot.add_to_conversation(
                "Search results:" + search_results,
                "system",
                convo_id="default",
            )
            if args.no_stream:
                print(chatbot.ask(prompt, "user", convo_id="default"))
            else:
                for query in chatbot.ask_stream(prompt):
                    print(query, end="", flush=True)
        else:
            if args.no_stream:
                print(chatbot.ask(prompt, "user"))
            else:
                for query in chatbot.ask_stream(prompt):
                    print(query, end="", flush=True)
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit()