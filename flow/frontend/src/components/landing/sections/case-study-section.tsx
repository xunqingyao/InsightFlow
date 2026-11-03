import Link from "next/link";

import { Card } from "@/components/ui/card";
import { pathOfThread } from "@/core/threads/utils";
import { cn } from "@/lib/utils";

import { Section } from "../section";

export function CaseStudySection({ className }: { className?: string }) {
  const caseStudies = [
    {
      threadId: "7cfa5f8f-a2f8-47ad-acbd-da7137baf990",
      title: "2026年智能体趋势与机遇预测",
      description: "创建深度研究报告网页，预测2026年智能体技术趋势和机遇。",
    },
    {
      threadId: "4f3e55ee-f853-43db-bfb3-7d1a411f03cb",
      title: "基于《傲慢与偏见》生成视频",
      description: "搜索小说《傲慢与偏见》中的特定场景，然后根据场景生成视频和参考图片。",
    },
    {
      threadId: "21cfea46-34bd-4aa6-9e1f-3009452fbeb9",
      title: "哆啦A梦解释MOE架构",
      description: "生成哆啦A梦漫画，向对AI感兴趣的青少年解释MOE架构。",
    },
    {
      threadId: "ad76c455-5bf9-4335-8517-fc03834ab828",
      title: "泰坦尼克号数据集探索性分析",
      description: "探索泰坦尼克号数据集，识别影响生存率的关键因素并生成可视化图表。",
    },
    {
      threadId: "d3e5adaf-084c-4dd5-9d29-94f1d6bccd98",
      title: "观看Y Combinator视频并进行深度研究",
      description: "观看指定的Y Combinator视频，对YC给技术创业者的建议进行深度研究。",
    },
    {
      threadId: "3823e443-4e2b-4679-b496-a9506eae462b",
      title: "收集并总结李飞飞博士的播客",
      description: "收集李飞飞博士过去6个月的所有播客出演，并整理成综合报告。",
    },
  ];
  return (
    <Section
      className={className}
      title="案例展示"
      subtitle="看看 DeerFlow 是如何被使用的"
    >
      <div className="container-md mt-8 grid grid-cols-1 gap-4 px-20 md:grid-cols-2 lg:grid-cols-3">
        {caseStudies.map((caseStudy) => (
          <Link
            key={caseStudy.title}
            href={pathOfThread(caseStudy.threadId) + "?mock=true"}
            target="_blank"
          >
            <Card className="group/card relative h-64 overflow-hidden">
              <div
                className="absolute inset-0 z-0 bg-cover bg-center bg-no-repeat transition-all duration-300 group-hover/card:scale-110 group-hover/card:brightness-90"
                style={{
                  backgroundImage: `url(/images/${caseStudy.threadId}.jpg)`,
                }}
              ></div>
              <div
                className={cn(
                  "flex h-full w-full translate-y-[calc(100%-60px)] flex-col items-center",
                  "transition-all duration-300",
                  "group-hover/card:translate-y-[calc(100%-128px)]",
                )}
              >
                <div
                  className="flex w-full flex-col p-4"
                  style={{
                    background:
                      "linear-gradient(to bottom, rgba(0, 0, 0, 0) 0%, rgba(0, 0, 0, 1) 100%)",
                  }}
                >
                  <div className="flex flex-col gap-2">
                    <h3 className="flex h-14 items-center text-xl font-bold text-shadow-black">
                      {caseStudy.title}
                    </h3>
                    <p className="box-shadow-black overflow-hidden text-sm text-white/85 text-shadow-black">
                      {caseStudy.description}
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </Link>
        ))}
      </div>
    </Section>
  );
}
