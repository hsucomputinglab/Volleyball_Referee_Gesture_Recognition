# Read the input data from the text file
input_filename = 'volleyball_annotation.txt'
output_filename = 'output.txt'

# Dictionary to store actions and their corresponding times
action_times = {}

# Read the input file
with open(input_filename, 'r') as file:
    for line in file:
        parts = line.strip().split()
        start_time = int(parts[0])
        end_time = int(parts[1])
        action = parts[2]
        
        # Add the times to the corresponding action in the dictionary
        if action not in action_times:
            action_times[action] = []
        action_times[action].extend([start_time, end_time])

# Write the output to a new text file
with open(output_filename, 'w') as file:
    for action, times in action_times.items():
        file.write(f'{action} = {times}\n')
