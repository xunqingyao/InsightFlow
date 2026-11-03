"use client";

import {
  AnimatedSpan,
  Terminal,
  TypingAnimation,
} from "@/components/ui/terminal";

import { Section } from "../section";

export function SandboxSection({ className }: { className?: string }) {
  return (
    <Section
      className={className}
      title="智能体运行环境"
      subtitle={
        <p>
          我们为 DeerFlow 提供了一台"电脑"，可以执行命令、管理文件、运行长时间任务 —— 全部在安全的 Docker 沙箱中
        </p>
      }
    >
      <div className="mt-8 flex w-full max-w-6xl flex-col items-center gap-12 lg:flex-row lg:gap-16">
        {/* Left: Terminal */}
        <div className="w-full flex-1">
          <Terminal className="h-[360px] w-full">
            {/* Scene 1: Build a Game */}
            <TypingAnimation>$ cat requirements.txt</TypingAnimation>
            <AnimatedSpan delay={800} className="text-zinc-400">
              pygame==2.5.0
            </AnimatedSpan>

            <TypingAnimation delay={1200}>
              $ pip install -r requirements.txt
            </TypingAnimation>
            <AnimatedSpan delay={2000} className="text-green-500">
              ✔ 已安装 pygame
            </AnimatedSpan>

            <TypingAnimation delay={2400}>
              $ write game.py --lines 156
            </TypingAnimation>
            <AnimatedSpan delay={3200} className="text-blue-500">
              ✔ 已写入 156 行
            </AnimatedSpan>

            <TypingAnimation delay={3600}>
              $ python game.py --test
            </TypingAnimation>
            <AnimatedSpan delay={4200} className="text-green-500">
              ✔ 所有精灵已加载
            </AnimatedSpan>
            <AnimatedSpan delay={4500} className="text-green-500">
              ✔ 物理引擎正常
            </AnimatedSpan>
            <AnimatedSpan delay={4800} className="text-green-500">
              ✔ 60 FPS 稳定
            </AnimatedSpan>

            {/* Scene 2: Data Analysis */}
            <TypingAnimation delay={5400}>
              $ curl -O sales-2024.csv
            </TypingAnimation>
            <AnimatedSpan delay={6200} className="text-zinc-400">
              已下载 12.4 MB
            </AnimatedSpan>
          </Terminal>
        </div>

        {/* Right: Description */}
        <div className="w-full flex-1 space-y-6">
          <div className="space-y-4">
            <p className="text-sm font-medium tracking-wider text-purple-400 uppercase">
              开源
            </p>
            <h2 className="text-4xl font-bold tracking-tight lg:text-5xl">
              <a
                href="https://github.com/agent-infra/sandbox"
                target="_blank"
                rel="noopener noreferrer"
              >
                AIO 沙箱
              </a>
            </h2>
          </div>

          <div className="space-y-4 text-lg text-zinc-400">
            <p>
              我们推荐使用{" "}
              <a
                href="https://github.com/agent-infra/sandbox"
                className="underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                All-in-One Sandbox
              </a>{" "}
              —— 将浏览器、Shell、文件系统、MCP 和 VSCode Server 整合在单个 Docker 容器中。
            </p>
          </div>

          {/* Feature Tags */}
          <div className="flex flex-wrap gap-3 pt-4">
            <span className="rounded-full border border-zinc-800 bg-zinc-900 px-4 py-2 text-sm text-zinc-300">
              隔离环境
            </span>
            <span className="rounded-full border border-zinc-800 bg-zinc-900 px-4 py-2 text-sm text-zinc-300">
              安全可靠
            </span>
            <span className="rounded-full border border-zinc-800 bg-zinc-900 px-4 py-2 text-sm text-zinc-300">
              持久化
            </span>
            <span className="rounded-full border border-zinc-800 bg-zinc-900 px-4 py-2 text-sm text-zinc-300">
              可挂载文件系统
            </span>
            <span className="rounded-full border border-zinc-800 bg-zinc-900 px-4 py-2 text-sm text-zinc-300">
              长时间运行
            </span>
          </div>
        </div>
      </div>
    </Section>
  );
}
