import re
from collections import OrderedDict
import traceback
import copy

class ReOrderSummary:
    def __init__(self, merge_regular, key_positions) -> None:
        with open(merge_regular, encoding='utf8') as f:
            self.key_merge_regular = {line.split('\t')[0]:line.split('\t')[1].strip() for line in f.readlines()}
        ordered_keys = OrderedDict()
        with open(key_positions, 'r', encoding='utf-8')  as f:
            for line in f.readlines():
                ordered_keys[line.strip()] = ''
        self.ordered_keys = ordered_keys

    def post_process_abs(self, all_abstract):
        """
        摘要后处理，对齐过滤
        1.按照给定的key序列排序；
        2.去重：根据key-value去重；根据子串去重合并；
        3.对于时间保留最后一个，前面删除；对于主要症状保留第一个，其他的插入作为伴随症状
        :param all_abstract:
        :return:
        """
        try:
            ordered_keys = copy.deepcopy(self.ordered_keys)
            conflict_main_symptom_keys = []
            time_to_accompany_order = {}
            abs_list = all_abstract.split("\n")
            for abs in abs_list:
                if not abs.strip() or not re.search(r':|：', abs):
                    continue
                k, v = re.split(r':|：', abs, maxsplit=1)
                if k not in ordered_keys:
                    # 待优化
                    print('删除键：', abs)
                    continue

                # 取{字符串_S, 字符串_M}两种值，S代表单值，后边覆盖前面，M代表多值，要合并
                merge_type = self.key_merge_regular[re.sub(r'\d+', '1', k)] 
                is_main_symptom = '主要症状.症状术语' in k
                is_not_main_symptom = not is_main_symptom

                # 主要症状处理逻辑
                if is_main_symptom:
                    if not ordered_keys[k]:
                        ordered_keys[k] = v
                    else:
                        # 冲突的主要症状要转换为伴随症状，需考虑序号，先暂存起来
                        conflict_main_symptom_keys.append((k,v))
                    continue
                # 其它键处理逻辑
                else:
                    if merge_type.endswith('_S'):
                        ordered_keys[k] = v
                    elif merge_type.endswith('_M'):
                        if v not in ordered_keys[k]:
                            ordered_keys[k] = f'{ordered_keys[k]},{v}'
                    # 需获取伴随症状序号，用以主要症状的键的更新
                    if '伴随症状' in k:
                        k_fields = k.split('.')
                        time_order = '.'.join(k_fields[:2])
                        last_order = time_to_accompany_order.get(time_order, 0)
                        time_to_accompany_order[time_order] = max(int(k_fields[2][-1]), last_order)
            for confict_k, v in conflict_main_symptom_keys:
                time_order = '.'.join(confict_k.split('.')[:2])
                new_accompany_order = time_to_accompany_order[time_order] + 1
                new_key = confict_k.replace('主要症状', f'伴随症状{new_accompany_order}')
                time_to_accompany_order[time_order] = new_accompany_order
                ordered_keys[new_key] = v

            result = [f'{k}:{v}' for k,v in ordered_keys.items() if v.strip()]
        except:
            traceback.print_exc()
            return all_abstract
        return "\n".join(result)

if __name__ == '__main__':
    all_abstract = r'''
现病史.时间1.主要症状.症状术语:下腹部疼痛
现病史.时间1.发生时间:3个月
现病史.时间1.主要症状.症状术语:后背疼痛
现病史.时间1.主要症状.性质程度:间断性疼痛
初步诊断.疾病名称1:后背痛待查
现病史.时间1.诊治经过.医疗机构名称:肝脏内科
现病史.时间1.主要症状.症状术语:肝区疼痛
现病史.时间1.诊治经过.检查检验项目及结果:肝脏B超、血检
现病史.时间1.诊治经过.检查检验项目及结果:嗜酸性粒细胞升高
现病史.时间1.诊治经过.检查检验项目及结果:嗜酸性粒细胞计数升高
现病史.时间1.诊治经过.检查检验项目及结果:嗜酸性粒细胞计数升高
现病史.时间1.主要症状.症状术语:腹痛
现病史.时间1.发生时间:2年
现病史.时间1.伴随症状1.症状术语:早起时肝区疼痛
现病史.时间1.伴随症状1.特点:有噎着感
现病史.时间1.伴随症状1.症状术语:吞咽困难
现病史.时间1.伴随症状1.症状术语:放射性疼痛
现病史.时间1.伴随症状1.特点:空腹时出现
现病史.时间1.伴随症状1.特点:疼痛可自行缓解
现病史.时间1.伴随症状1.症状术语:乏力
现病史.时间1.伴随症状1.症状术语:大汗
现病史.时间1.伴随症状1.症状术语:蜷曲体位可缓解
现病史.时间1.诊治经过.病情转归:疼痛可自行缓解
现病史.时间1.诊治经过.病情转归:发作频率不高
现病史.时间1.发生时间:2年
现病史.时间1.发生频率:间断发生
现病史.时间1.发生时间:1个月
现病史.时间1.发生时间:1到2次/月
现病史.时间1.发生时间:1-2次/月
现病史.时间1.阴性症状:无熬夜、劳累等诱因
现病史.时间1.主要症状.性质程度:疼痛向右侧肋部、背部放射
现病史.时间1.主要症状.性质程度:疼痛可放射至右侧后背
现病史.时间1.伴随症状1.症状术语:胸部中间疼痛
现病史.时间1.阴性症状:无皮疹
现病史.一般情况.大小便:每日一次大便
现病史.一般情况.大小便:大便正常
现病史.一般情况.大小便:大便颜色正常
既往史.其他信息:无心脏、肝脏、血压、血糖、血脂、心脑血管等其他病
既往史.其他信息:无高血压、糖尿病、高血脂、心脑血管等疾病史。
既往史.药物过敏:某些药物过敏
既往史.药物过敏:西西里情
既往史.药物过敏:茶
既往史.其他信息:有过敏性鼻炎
既往史.疾病史1.疾病名称:过敏性鼻炎
既往史.其他信息:近2年无荨麻疹
既往史.疾病史1.疾病名称:头孢过敏
婚育史月经史.月经:月经周期14天，末次月经2023/6/14
婚育史月经史.月经:末次月经2023-6-24
婚育史月经史.月经:已绝经
现病史.时间1.主要症状.性质程度:疼痛位于上腹部偏左
现病史.时间1.主要症状.性质程度:疼痛有一个钝点
现病史.时间1.诊治经过.检查检验项目及结果:肝脏b超未见异常
处理意见.检查项目名称:胃肠镜
处理意见.其他建议:半年后完善胃肠镜检查
现病史.时间1.发生时间:2年
现病史.时间1.发生时间:1个月1-2次
现病史.时间1.诊治经过.检查检验项目及结果:心脏未见异常
现病史.时间1.诊治经过.检查检验项目及结果:胃肠镜检查无病变
处理意见.其他建议:观察
处理意见.其他建议:如症状频繁，则行胃肠镜检查
    '''
    app = ReOrderSummary(
        merge_regular=r'scripts\training\all_summary_keys\merge_regular.tsv',
        key_positions=r'scripts\training\all_summary_keys\key_position.txt'
    )
    print(app.post_process_abs(all_abstract))