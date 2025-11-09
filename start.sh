#!/bin/bash

# =============================================================================
# Real Estate Multi-Agent System - Startup Script
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop."
        exit 1
    fi
    print_success "Docker is installed"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed."
        exit 1
    fi
    print_success "Docker Compose is installed"
    
    # Check .env file
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_info "Creating .env from .env.example..."
        if [ -f .env.example ]; then
            cp .env.example .env
            print_warning "Please edit .env file with your API keys and settings"
            print_info "Run: nano .env"
            exit 1
        else
            print_error ".env.example not found!"
            exit 1
        fi
    fi
    print_success ".env file exists"
}

# Check required directories
check_directories() {
    print_header "Checking Directories"
    
    directories=("data" "data/images" "data/certificates" "data/reports" "models")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            print_warning "$dir does not exist. Creating..."
            mkdir -p "$dir"
        fi
        print_success "$dir exists"
    done
}

# Check required files
check_files() {
    print_header "Checking Required Files"
    
    # Check model files
    if [ ! -f "models/floorplan_checkpoint.pth" ]; then
        print_warning "Floorplan checkpoint not found: models/floorplan_checkpoint.pth"
        print_info "Place your model checkpoint in the models/ directory"
    else
        print_success "Floorplan checkpoint found"
    fi
    
    if [ ! -f "models/room_classifier.pkl" ]; then
        print_warning "Room classifier not found: models/room_classifier.pkl"
        print_info "Place your classifier in the models/ directory"
    else
        print_success "Room classifier found"
    fi
}

# Build services
build_services() {
    print_header "Building Docker Images"
    
    print_info "This may take several minutes on first run..."
    if docker-compose build; then
        print_success "Docker images built successfully"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
}

# Start services
start_services() {
    print_header "Starting Services"
    
    if docker-compose up -d; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Wait for services
wait_for_services() {
    print_header "Waiting for Services to be Ready"
    
    print_info "Waiting for PostgreSQL..."
    timeout 60 bash -c 'until docker-compose exec -T postgres pg_isready -U postgres; do sleep 2; done' || {
        print_error "PostgreSQL failed to start"
        docker-compose logs postgres
        exit 1
    }
    print_success "PostgreSQL is ready"
    
    print_info "Waiting for Qdrant..."
    timeout 60 bash -c 'until curl -sf http://localhost:6333/health > /dev/null; do sleep 2; done' || {
        print_error "Qdrant failed to start"
        docker-compose logs qdrant
        exit 1
    }
    print_success "Qdrant is ready"
    
    print_info "Waiting for Backend..."
    timeout 120 bash -c 'until curl -sf http://localhost:8000/health > /dev/null; do sleep 3; done' || {
        print_error "Backend failed to start"
        docker-compose logs backend
        exit 1
    }
    print_success "Backend is ready"
    
    print_info "Waiting for Frontend..."
    timeout 60 bash -c 'until curl -sf http://localhost:8501/_stcore/health > /dev/null; do sleep 2; done' || {
        print_warning "Frontend might not be ready yet (this is okay)"
    }
    print_success "Frontend is starting"
}

# Show status
show_status() {
    print_header "Service Status"
    docker-compose ps
}

# Show access information
show_access_info() {
    print_header "Access Information"
    echo ""
    print_success "Frontend (Streamlit):  http://localhost:8501"
    print_success "Backend API:           http://localhost:8000"
    print_success "API Documentation:     http://localhost:8000/docs"
    print_success "Qdrant Dashboard:      http://localhost:6333/dashboard"
    echo ""
    print_info "To view logs: docker-compose logs -f"
    print_info "To stop:      docker-compose down"
    echo ""
}

# Main execution
main() {
    clear
    print_header "Real Estate Multi-Agent System - Startup"
    echo ""
    
    # Run checks
    check_prerequisites
    check_directories
    check_files
    
    # Ask for confirmation
    echo ""
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Startup cancelled"
        exit 0
    fi
    
    # Build and start
    build_services
    start_services
    wait_for_services
    
    # Show results
    show_status
    show_access_info
    
    print_header "System Ready!"
    print_success "All services are running successfully!"
    echo ""
}

# Run main function
main