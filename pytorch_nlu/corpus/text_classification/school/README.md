# 数据集
##数据来源: 
 - url: https://github.com/FBI1314/textClassification/tree/master/multilabel_text_classfication/data
 - 详情: unknow, 来源未知, 多标签分类语料, 约22339语料, 7个类别.
    
##备注: 
这里训练、验证、测试集各只取了132个样例

##数据分析:
```bash

file = open("train.json", encoding="utf-8")
texts = file.readlines()


label_text = {}
for text in texts:
    text_json = eval(text.strip())
    label = text_json["label"]
    text = text_json["text"]
    if label not in label_text:
        label_text[label] = [text]
    else:
        if len(label_text[label]) < 10:
            label_text[label] += [text]

print(label_text)



id_to_label = {
"0": "校园活动",
"1": "文明礼仪",
"2": "文化课程",
"3": "课后活动",
"4": "性格情绪",
"5": "德育量化",
"6": "作息生活"}



train_data = {
 '0': ['体育课被批评', '挑战性问题', '设备现场无问题', '加错分抵消', '校级活动', '啪啪啪', '请课＂中＂', '三自班级', '专门评价陈屌丝的', '特别奖'],
 '1': ['个人卫生（指甲太长）', '专注之星季军奖励6分', '不戴红领巾校章', '个人卫生保持差', '小老师管理', '精神面貌好', '磨蹭', '优秀主持人', '语文作业完成质量高', '会话表演之星'],
 '2': ['单科考试95分以上', '表达清晰有想法', '语文80分以上', '小组回答问题', '小小写手', '生物作业优秀', '期中测试80-89', '按时完成作业改错', '英语听写全对', '成绩退步明显'],
 '3': ['下课乱跳大叫', '上课比较认真', '早到且迅速开始学习', '下位走动', '书皮没包没姓名签', '静校讲话', '语文家庭作业没完成', '书写干净整齐', '应付作业', '安静就餐不浪费'],
 '4': ['愤怒', '带情绪练琴', '自己的事情自己做', '神经三八很废', '情绪过度', '控制自己的情绪', '跟她妈生气', '管理好情绪', '情绪消极', '不吵不闹'],
 '5': ['秩序感优秀', '【好事】好人好事', '爱护班级卫生', '宿舍扣分', '上课乖', '乐于帮助他人', '文明有礼', '秩序之星', '教师助手', '尊师重道'],
 '6': ['做位险的游戏', '态度消极', '举手回答问题2次', '各位活动积极参与奖', '动作迅速有效', '懒洋洋慢吞吞', '晚修吵闹', '起床迅速', '老师小助手月奖励', '举手回答'],

 '3#2': ['作业不改', '作业书写较乱', '按要求完成语文作业', '优化认真辅导、批改。', '订正多次', '写字进步', '成绩进步或优秀', '语文作业完美', '堂听满分', '中文经典5分'],
 '1#3': ['自觉预习', '书写凌乱', '课前准备小标兵', '完成13号的语文作业', '卷面脏', '未做放学卫生值日', '今日工作突出', '没带美术用具', '扫地逃跑', '认真写1号作业'],
 '1#5': ['值日班长、卫生督导员', '榜样作用', '坐姿补端', '课堂合作愉快！', '爱劳动讲卫生', '清洁流动红旗', '扫地偷懒', '主动为班级加分', '献爱心积极分子', '上课期间穿拖鞋'],
 '3#5': ['连带纪律适当', '不文明说脏话', '给力小助手', '说脏话、起外号、打人', '履行职责', '遵守纪律三操认真', '发脾气破坏公物', '顶撞老师', '温柔又专心', '到校准时'],
 '5#2': ['带头读书', '善于分享交流', '班干部或组长嘉奖', '引领带动帮助同学', '文明发言会倾听', '数学小老师', '班级职责不到位', '好样的小老师', '分享好点子', '劳动积极工作负责'],
 '1#6': ['回答3次及以上', '回答3次', '音乐课积极发言', '运动会', '没值日', '主动练琴', '课前一支歌表现差', '发言小达人', '辛苦劳动', '语文课答问'],
 '3#6': ['政治缺交', '快乐晨读', '作业拖欠', '今日事今日毕作业完成', '主动帮助他人', '3点半作业拖拉没写完', '自习离位', '做与课堂无关事', '背诵不按时', '安全平台按时完成'],
 '3#1': ['积极背单词', '乱扔果皮纸屑', '主动做作业', '没有预习课文', '超额完成作业', '桌子没摆正', '配合打扫好班级卫生', '做清洁认真', '数学课表扬', '做卫生认真'],
 '4#3': ['大课间纪律差', '数学课堂注意力集中', '自习课纪律差大吵大闹', '字写得好'],
 '1#2': ['预习小能手', '大型集会说话', '不卫生学习差有神经病', '答问声音响亮', '我会领读', '参加辩论赛', '积极发言，善于表达', '举手是我的日常操作', '课堂积极动脑举手答题', '板演'],
 '5#6': ['上课说话屡教不改', '就寝时讲话吵闹', '两操检查', '上课玩东西，不认真', '站队做到快、静、齐', '做作业讲话', '队列中讲话', '上课扭头', '打架斗殴', '餐厅安静回位'],
 '2#6': ['能运用学过的词语句子', '作业速度太慢', '唱歌', '积极思考有自己的想法', '回答问题积极声音洪亮', '作业速度快', '回答问题声音洪亮', '数学课回答声音洪亮', '物理作业及订正认真',
         '回答问题声音洪亮干脆'], '4#5': ['乱发脾气', '情绪失控-说不文明话'], '4#1': ['吃饭安静有序'], '4#6': ['玩得很开心', '过度兴奋', '发生肢体冲突'],
 '4#2': ['关心班级']}




val_data = {
 '0': ['我的眼睛', '大神经', '老师特别表扬', '上周表现良好'],
 '1': ['小小图书管理员', '班级打扫卫生', '奖励', '教室卫生达人', '不及时打扫卫生', '升旗小能手', '餐点不挑食不浪费', '二级巨星', '小组表现不好', '歌曲变歌词小能手'],
 '2': ['课前提问回答不完整', '提出质疑', '作业完成情况优秀', '我会算', '看拼音写词语纸展览', '作业完成不好', '80分以下', '读得好', '非常优秀', '考试低分'],
 '3': ['运动会打架', '认真完成聪明题', '静校未及时，说话', '准时返校', '上课打闹，屡教不改', '课堂表现类', '午餐后自我管理差', '部分作业没完成', '晚拖讲话', '国庆作业'],
 '4': ['发火，生气', '管理情绪', '开朗乐观'],
 '5': ['善良，有孝心', '说脏话起外号', '携带零食', '队列安静有序', '课堂集体评比第3名', '温柔听话', '雷同作业', '集合速度快', '喜欢偷偷说别人', '关心、帮助他人'],
 '6': ['动作拖拉慢吞吞', '跑步第一', '心不在焉', '9点半之前没上床睡觉', '运动会第一名', '做事拖拉慢吞吞', '太累了', '体育运动棒', '课前准备没有拿书出来', '课堂上举手积极发言'],
 '3#5': ['课堂搞东西，没认真听', '不团结', '化学老师的得力助手', '未完成老师布置的作业', '追跑打闹，打骂同学'],
 '3#2': ['测验90-94.A-', '长江作业未交', '美术作业棒棒哒', '学霸笔记', '主动学习看书', '作文不合格', '单元考试进步很大', '订正过关', '作文修改后得优加', '期中考前十名'],
 '1#3': ['内务脏乱、静校慢'],
 '3#6': ['老师不在，讲话', '动作迅速不讲话', '干扰同桌', '大课间good', '按时到校值日', '做事不安静', '到班不及时', '不写笔记', '考试的时候讲话', '按时上交材料'],
 '4#5': ['善良可爱美丽大方'],
 '5#6': ['乱起哄', '不好好做眼保健操'], '1#5': ['有错不改脾气倔', '流动红旗'],
 '1#2': ['创新表演'],
 '2#6': ['聪明活泼']}

```