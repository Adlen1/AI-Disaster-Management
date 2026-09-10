import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = { title: "Algeria Wildfire Risk", description: "Clear next-day commune-level wildfire intelligence for Algeria." };
export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) { return <html lang="en" data-theme="light"><body>{children}</body></html>; }
