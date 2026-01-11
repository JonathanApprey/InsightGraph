import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Set output file tracing root to the frontend directory
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;
