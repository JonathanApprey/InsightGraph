import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ 
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "InsightGraph - Agentic RAG Platform",
  description: "Smart Document Assistant powered by LangGraph agents that doesn't just retrieve—it thinks.",
  keywords: ["RAG", "LangGraph", "AI", "Document Assistant", "Generative AI"],
  authors: [{ name: "Jonathan Ekowo-Apprey" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
