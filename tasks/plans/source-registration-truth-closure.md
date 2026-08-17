# Q64 Source registration truth闭合实施计划

状态：`queued`。稳定source ownership和library依赖由18号设计文档拥有；动态状态只看`tasks/progress.md`。Q51.Core、Q62、Q63
先删除各自已确认的旧source/test island，Q64再关闭全仓checker覆盖缺口。

当前`tools/check_source_organization.py`只对少数目录执行active/dormant集合闭包，仓库其余C++ translation unit和test可能既未进入
CMake，也未在18号dormant owner表出现而仍获得green结果。旧optimization comparison “source contract”又反向读取源码marker，
把未注册旧search文件伪装成必须保留的当前资产。CMake、filesystem、手写allowlist和Python catalog因此不是同一事实源。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  repository source/test filesystem、current CMake target/test registration、generated declarations及18号明确的task-owned dormant迁移表。
- Current stage responsibility:
  建立repo-wide registration mirror：每个可编译translation unit和测试要么属于一个current CMake target/CTest/lit owner，要么由
  一个current编号任务明确记录唯一能力承接与删除门禁；其它文件fail closed并删除或注册，不能被隐式忽略。
- Output IR / files:
  不改变compiler IR或产品output；输出唯一source/test inventory检查、收敛后的CMake registration和最小必要task-owned dormant表。
- Downstream consumer:
  fresh configure/build、unit/lit/CTest、public link closure以及Q50/Q51/Q62后续删除门禁。
- User-level driver / named pipeline:
  current CMake configure和source-organization检查；不新增compiler CLI、pass或runtime mode。
- Explicit non-goals:
  不把所有dormant source机械加入build，不为通过检查创建空target/stub，不保留retired symbol/source marker test，
  不把archive或generated build tree当source registration。
- Done criteria:
  repo-wide所有C/C++ source与测试均有且只有一个active registration或current task-owned dormant disposition；新增未注册文件使检查
  失败；registered test不读取implementation symbol/source文本来保活旧实现；CMake/source/test mirror、fresh configure/build和
  direct organization tests通过。
```

## 施工规则

- filesystem枚举排除build/generated/archive后，与CMake实际target source、unit target、lit/CTest registration做双向核对；
  checker不能只维护几个目录的硬编码集合。
- dormant记录必须包含current task owner、独有能力和删除门禁；只有“以后可能有用”或source未进CMake不能进入allowlist。
- Q51.Core删除Rank/coordinated search、paired optimization catalog/source-marker test；Q62删除numeric umbrella与迁出managed dependency
  conformance；Q63删除NCC free-switch旧owner。Q64不为这些文件重新建立dormant例外。
- source与test同步：实现未注册但test读取marker、test未注册但被文档称为gate、public header可include但无link symbol都作为错误。
- organization checker只证明注册/owner事实，不把FileCheck、源码marker或catalog计数冒充功能正确性。

## 验证

- 为unregistered source、unregistered test、active+dormant overlap、unknown dormant owner和source-marker保活test提供checker负例；
- fresh configure后从实际CMake graph重算inventory，运行source organization、public header/link smoke和受影响direct tests；
- 不运行Q49.P/Q51长搜索、历史board资格或与source registration无关的全量integration。
