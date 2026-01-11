FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

# Build for production
RUN npm run build -- --webpack

# Expose port
EXPOSE 3000

# Start production server
CMD ["npm", "start"]
