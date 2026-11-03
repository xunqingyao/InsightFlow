"use client";

import MagicBento, { type BentoCardProps } from "@/components/ui/magic-bento";
import { cn } from "@/lib/utils";

import { Section } from "../section";

const COLOR = "#0a0a0a";
const features: BentoCardProps[] = [
  {
    color: COLOR,
    label: "上下文工程",
    title: "长/短期记忆",
    description: "智能体现在能更好地理解你",
  },
  {
    color: COLOR,
    label: "长时间任务",
    title: "规划与子任务",
    description: "提前规划、复杂推理、顺序或并行执行",
  },
  {
    color: COLOR,
    label: "可扩展",
    title: "技能与工具",
    description: "即插即用，甚至可以替换内置工具，打造你想要的智能体。",
  },

  {
    color: COLOR,
    label: "持久化",
    title: "带文件系统的沙箱",
    description: "读、写、运行 —— 像真正的电脑一样",
  },
  {
    color: COLOR,
    label: "灵活性",
    title: "多模型支持",
    description: "豆包、DeepSeek、OpenAI、Gemini 等",
  },
  {
    color: COLOR,
    label: "免费",
    title: "开源",
    description: "MIT 许可证，自托管，完全控制",
  },
];

export function WhatsNewSection({ className }: { className?: string }) {
  return (
    <Section
      className={cn("", className)}
      title="DeerFlow 2.0 新特性"
      subtitle="DeerFlow 现在正在从深度研究智能体演变为全栈超级智能体"
    >
      <div className="flex w-full items-center justify-center">
        <MagicBento data={features} />
      </div>
    </Section>
  );
}
