# =============================================================================
# Makefile for Real Estate Multi-Agent System
# =============================================================================

.PHONY: help setup start stop restart logs build clean status health

# Default target
.DEFAULT_GOAL := help

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

help: ## Show this help message
	@echo "$(BLUE)Real Estate Multi-Agent System - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

setup: ## Initial setup (create .env, directories)
	@echo "$(BLUE)Setting up environment...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env file. Please edit with your API keys.$(NC)"; \
	else \
		echo "$(YELLOW)⚠ .env already exists$(NC)"; \
	fi
	@mkdir -p data/images data/certificates data/reports models
	@echo "$(GREEN)✓ Created directories$(NC)"

build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	@docker-compose build
	@echo "$(GREEN)✓ Build complete$(NC)"

start: ## Start all services
	@echo "$(BLUE)Starting services...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@make status

stop: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	@docker-compose stop
	@echo "$(GREEN)✓ Services stopped$(NC)"

down: ## Stop and remove containers
	@echo "$(BLUE)Stopping and removing containers...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✓ Containers removed$(NC)"

restart: ## Restart all services
	@echo "$(BLUE)Restarting services...$(NC)"
	@docker-compose restart
	@echo "$(GREEN)✓ Services restarted$(NC)"

logs: ## Show logs (Ctrl+C to exit)
	@docker-compose logs -f

logs-backend: ## Show backend logs only
	@docker-compose logs -f backend

logs-frontend: ## Show frontend logs only
	@docker-compose logs -f frontend

status: ## Show service status
	@echo "$(BLUE)Service Status:$(NC)"
	@docker-compose ps
	@echo ""

health: ## Check service health
	@echo "$(BLUE)Health Checks:$(NC)"
	@echo -n "Backend:   "
	@curl -sf http://localhost:8000/health > /dev/null && echo "$(GREEN)✓ Healthy$(NC)" || echo "$(YELLOW)✗ Unhealthy$(NC)"
	@echo -n "Frontend:  "
	@curl -sf http://localhost:8501/_stcore/health > /dev/null && echo "$(GREEN)✓ Healthy$(NC)" || echo "$(YELLOW)✗ Unhealthy$(NC)"
	@echo -n "Qdrant:    "
	@curl -sf http://localhost:6333/health > /dev/null && echo "$(GREEN)✓ Healthy$(NC)" || echo "$(YELLOW)✗ Unhealthy$(NC)"
	@echo -n "PostgreSQL:"
	@docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1 && echo "$(GREEN)✓ Healthy$(NC)" || echo "$(YELLOW)✗ Unhealthy$(NC)"

shell-backend: ## Open shell in backend container
	@docker-compose exec backend bash

shell-postgres: ## Open PostgreSQL shell
	@docker-compose exec postgres psql -U postgres -d real_estate_db

backup-db: ## Backup PostgreSQL database
	@echo "$(BLUE)Backing up database...$(NC)"
	@docker-compose exec -T postgres pg_dump -U postgres real_estate_db > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Backup created$(NC)"

restore-db: ## Restore PostgreSQL database (usage: make restore-db FILE=backup.sql)
	@if [ -z "$(FILE)" ]; then \
		echo "$(YELLOW)Usage: make restore-db FILE=backup.sql$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restoring database from $(FILE)...$(NC)"
	@docker-compose exec -T postgres psql -U postgres real_estate_db < $(FILE)
	@echo "$(GREEN)✓ Database restored$(NC)"

clean: ## Remove all containers, volumes, and images
	@echo "$(YELLOW)⚠ WARNING: This will delete all data!$(NC)"
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@echo "$(BLUE)Cleaning up...$(NC)"
	@docker-compose down -v --rmi all
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

update: ## Pull latest code and rebuild
	@echo "$(BLUE)Updating system...$(NC)"
	@git pull origin main
	@docker-compose build --no-cache
	@docker-compose down
	@docker-compose up -d
	@echo "$(GREEN)✓ Update complete$(NC)"

test: ## Run tests (if available)
	@echo "$(BLUE)Running tests...$(NC)"
	@docker-compose exec backend pytest tests/ -v

dev: ## Start in development mode
	@echo "$(BLUE)Starting in development mode...$(NC)"
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

urls: ## Show access URLs
	@echo "$(BLUE)Access URLs:$(NC)"
	@echo "  Frontend:         http://localhost:8501"
	@echo "  Backend API:      http://localhost:8000"
	@echo "  API Docs:         http://localhost:8000/docs"
	@echo "  Qdrant Dashboard: http://localhost:6333/dashboard"