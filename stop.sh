#!/bin/bash

# =============================================================================
# Real Estate Multi-Agent System - Stop/Cleanup Script
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Menu
show_menu() {
    clear
    print_header "Real Estate Multi-Agent System - Shutdown Menu"
    echo ""
    echo "1) Stop services (keep data)"
    echo "2) Stop and remove containers"
    echo "3) Full cleanup (WARNING: removes all data!)"
    echo "4) View logs before stopping"
    echo "5) Cancel"
    echo ""
}

# Stop services
stop_services() {
    print_header "Stopping Services"
    docker-compose stop
    print_success "Services stopped"
}

# Remove containers
remove_containers() {
    print_header "Removing Containers"
    docker-compose down
    print_success "Containers removed"
}

# Full cleanup
full_cleanup() {
    print_warning "This will remove:"
    echo "  - All containers"
    echo "  - All volumes (database data, vector store data)"
    echo "  - Docker images"
    echo ""
    read -p "Are you absolutely sure? Type 'yes' to confirm: " -r
    echo ""
    
    if [[ $REPLY == "yes" ]]; then
        print_header "Full Cleanup"
        
        print_info "Stopping and removing containers..."
        docker-compose down -v
        
        print_info "Removing images..."
        docker-compose down --rmi all
        
        print_success "Full cleanup completed"
        print_warning "All data has been removed!"
    else
        print_info "Cleanup cancelled"
    fi
}

# View logs
view_logs() {
    print_header "Service Logs"
    echo "Press Ctrl+C to exit logs"
    echo ""
    sleep 2
    docker-compose logs -f
}

# Main menu loop
main() {
    while true; do
        show_menu
        read -p "Select option (1-5): " choice
        echo ""
        
        case $choice in
            1)
                stop_services
                break
                ;;
            2)
                remove_containers
                break
                ;;
            3)
                full_cleanup
                break
                ;;
            4)
                view_logs
                ;;
            5)
                print_info "Operation cancelled"
                exit 0
                ;;
            *)
                print_warning "Invalid option. Please select 1-5."
                sleep 2
                ;;
        esac
    done
    
    echo ""
    print_success "Done!"
}

# Check if docker-compose is running
if ! docker-compose ps | grep -q "Up"; then
    print_warning "No services appear to be running"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

main