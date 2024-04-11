import os
import bs4
import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


def request_get_with_sleep(url, sleep_time=3):
    chrome_driver_path = os.path.abspath('./resource/chromedriver/chromedriver.exe')
    options = webdriver.ChromeOptions()
    options.add_argument('--log-level=3')  # 将 Chrome 浏览器的日志等级设置为 3，表示只输出错误信息，不输出运行信息
    options.add_experimental_option('useAutomationExtension', False)
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_argument('--ignore-certificate-errors')
    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(sleep_time)
    return driver


def get_dom():
    symptom_file = './resource/symptoms.csv'

    root = "https://www.mayoclinic.org"
    main_ref = root + "/symptom-checker/select-symptom/itt-20009075"
    html = request_get_with_sleep(main_ref).page_source
    dom = bs4.BeautifulSoup(html, 'html.parser')

    adult_symptom_ele = dom.find('div', attrs={'class': 'adult'})
    children_symptom_ele = dom.find('div', attrs={'class': 'child'})
    full_list = adult_symptom_ele.findAll('a') + children_symptom_ele.findAll('a')

    symptom_href_dict = {}
    for item in full_list:
        if hasattr(item, 'name') and item.name == 'a':
            key = item.text
            href = item.attrs['href']
            symptom_href_dict[key] = href

    symptom_dict = dict()
    for key in symptom_href_dict:
        symptom_dict[key] = dict()
        print('{}: {}'.format(key, symptom_href_dict[key]))
        symptom_href = root + symptom_href_dict[key]
        html = request_get_with_sleep(symptom_href).page_source
        dom = bs4.BeautifulSoup(html, 'html.parser')

        factor_ele = dom.find('div', attrs={'class': 'form'})
        factor_group_list = factor_ele.findAll('div', attrs={'class': 'frm_item'})
        for factor_group in factor_group_list:
            if not (hasattr(factor_group, 'name') and factor_group.name == 'div'):
                continue
            second_level_key = factor_group.find('legend').text
            symptom_dict[key][second_level_key] = list()
            factor_list = factor_group.findAll('li')
            for factor in factor_list:
                if not (hasattr(factor, 'name') and factor.name == 'li'):
                    continue
                factor = factor.find('label').text
                symptom_dict[key][second_level_key].append(factor)
                print('{}, {}, {}'.format(key, second_level_key, factor))

    data_to_write = [['symptom', 'factor_group', 'factor']]
    for key in symptom_dict:
        for second_key in symptom_dict[key]:
            for factor in symptom_dict[key][second_key]:
                data_to_write.append([key, second_key, factor])
    with open(symptom_file, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)
    return symptom_dict


def main():
    dom = get_dom()
    print(dom)


if __name__ == "__main__":
    main()
