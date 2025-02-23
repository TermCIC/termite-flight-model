import os

def project_manager():
    # Define the base projects directory
    projects_dir = "projects"

    # Check if the 'projects' folder exists; if not, create it.
    if not os.path.exists(projects_dir):
        os.makedirs(projects_dir)
        print(f"Created '{projects_dir}' folder.")
        project_manager()
    else:
        print(f"'{projects_dir}' folder already exists.")

    # List existing project directories within 'projects'
    existing_projects = [d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))]
    
    if existing_projects:
        print("\nExisting projects:")
        for idx, project in enumerate(existing_projects, start=1):
            print(f"  {idx}. {project}")
    else:
        print("\nNo existing projects found.")

    # Prompt user for selecting an existing project or creating a new one
    choice = input("\nEnter the project number to select, or type 'n' to create a new project: ").strip()

    if choice.lower() == 'n':
        # Creating a new project
        project_name = input("Enter new project name: ").strip()
        project_path = os.path.join(projects_dir, project_name)

        if os.path.exists(project_path):
            print(f"Project '{project_name}' already exists.")
            return project_path

        os.makedirs(project_path)
        print(f"Created new project folder: {project_path}")

        # List of required subfolders
        subfolders = [
            "db",
            "ensemble_weights",
            "evaluation_results",
            "models",
            "output",
            "splitted data",
            "threshold_results"
        ]
        # Create each subfolder within the new project folder
        for folder in subfolders:
            subfolder_path = os.path.join(project_path, folder)
            os.makedirs(subfolder_path, exist_ok=True)
            print(f"Created subfolder: {subfolder_path}")

        return project_path

    else:
        # Attempt to convert input to a project index
        try:
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(existing_projects):
                selected_project = existing_projects[selected_index]
                project_path = os.path.join(projects_dir, selected_project)
                print(f"Selected project: {selected_project}")
                return project_path
            else:
                print("Invalid project number.")
                return None
        except ValueError:
            print("Invalid input. Please enter a valid project number or 'n' to create a new project.")
            return None

if __name__ == "__main__":
    project_manager()
