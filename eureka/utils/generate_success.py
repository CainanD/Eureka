#!/usr/bin/env python3
"""
Script to transform CartpoleGPT code by replacing compute_success function
body with compute_reward function body and updating the method calls accordingly.
"""

import re


def replace_exterior_compute_reward(code_string, replacement_string):
    """
    Remove the exterior compute_reward function (including @torch.jit.script decorator)
    and replace it with a given string. Also updates the interior compute_reward method call
    to match the new signature.
    
    Args:
        code_string: String containing the Python code to transform
        replacement_string: String to replace the compute_reward function with
        
    Returns:
        Transformed code string with compute_reward function replaced
    """
    
    import ast
    
    lines = code_string.split('\n')
    
    # Find compute_reward function (exterior one, after the class)
    reward_start_idx = -1
    reward_end_idx = -1
    reward_def_idx = -1
    
    # Track if we've passed the class definition
    in_class = False
    class_ended = False
    
    for i, line in enumerate(lines):
        # Track class boundaries
        if line.strip().startswith('class ') and 'VecTask' in line:
            in_class = True
        elif in_class and line and not line[0].isspace() and not line.strip().startswith('#'):
            # We've left the class (non-indented, non-empty, non-comment line)
            if not line.strip().startswith('class'):
                class_ended = True
                in_class = False
        
        # Find compute_reward (exterior one, should come after class)
        if 'def compute_reward(' in line and class_ended:
            reward_def_idx = i
            # Look backwards for @torch.jit.script
            for j in range(i-1, max(0, i-10), -1):
                if '@torch.jit.script' in lines[j]:
                    reward_start_idx = j
                    break
            if reward_start_idx == -1:
                reward_start_idx = i
            
            # Find the end of compute_reward function
            for j in range(i+1, len(lines)):
                line_stripped = lines[j].lstrip()
                if line_stripped and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                    if line_stripped.startswith('def ') or line_stripped.startswith('class ') or \
                       line_stripped.startswith('from ') or line_stripped.startswith('import ') or \
                       line_stripped.startswith('@'):
                        reward_end_idx = j
                        break
            
            if reward_end_idx == -1:
                reward_end_idx = len(lines)
            break
    
    if reward_start_idx == -1:
        print("Warning: exterior compute_reward function not found")
        return code_string
    
    # Build the new code with replacement
    new_lines = (
        lines[:reward_start_idx] +  # Everything before compute_reward
        [replacement_string] +  # Replacement string
        lines[reward_end_idx:]  # Everything after compute_reward
    )
    
    result = '\n'.join(new_lines)
    
    # Try to extract signature from replacement_string and update interior call
    try:
        module = ast.parse(replacement_string)
        function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        if function_defs and function_defs[0].name == 'compute_reward':
            function_def = function_defs[0]
            args = [arg.arg for arg in function_def.args.args]
            new_call = 'compute_reward(self.' + ', self.'.join(args) + ')'
            
            # Pattern to find interior compute_reward call - more flexible
            # Matches: anything = compute_reward(...)
            interior_call_pattern = r'([\w\[\]:,\s]+)\s*=\s*compute_reward\s*\([^)]*\)'
            
            def replace_call(match):
                lhs = match.group(1).strip()
                return f"{lhs} = {new_call}"
            
            result = re.sub(interior_call_pattern, replace_call, result)
    except:
        pass  # If parsing fails, just return the result without updating the call
    
    # Clean up any extra blank lines
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    return result


def transform_code(code_string):
    """
    Transform the code by:
    1. Finding the exterior compute_reward function and extracting its signature and body
    2. Replacing the compute_success function's signature and body with compute_reward's
    3. Keeping both exterior functions
    4. Updating the compute_reward method call to use the new format with self-prefixed arguments
    
    Args:
        code_string: String containing the Python code to transform
        
    Returns:
        Transformed code string
    """
    
    import ast
    
    lines = code_string.split('\n')
    
    # Find compute_success function
    success_start_idx = -1
    success_end_idx = -1
    success_def_idx = -1
    
    # Find compute_reward function (exterior one, after the class)
    reward_start_idx = -1
    reward_end_idx = -1
    reward_def_idx = -1
    
    # Track if we've passed the class definition
    in_class = False
    class_ended = False
    
    for i, line in enumerate(lines):
        # Track class boundaries
        if line.strip().startswith('class ') and 'VecTask' in line:
            in_class = True
        elif in_class and line and not line[0].isspace() and not line.strip().startswith('#'):
            # We've left the class (non-indented, non-empty, non-comment line)
            if not line.strip().startswith('class'):
                class_ended = True
                in_class = False
        
        # Find compute_success (should be after class)
        if 'def compute_success(' in line and class_ended:
            success_def_idx = i
            # Look backwards for @torch.jit.script
            for j in range(i-1, max(0, i-10), -1):
                if '@torch.jit.script' in lines[j]:
                    success_start_idx = j
                    break
            if success_start_idx == -1:
                success_start_idx = i
            
            # Find the end of compute_success function
            for j in range(i+1, len(lines)):
                line_stripped = lines[j].lstrip()
                if line_stripped and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                    if line_stripped.startswith('def ') or line_stripped.startswith('class ') or \
                       line_stripped.startswith('from ') or line_stripped.startswith('import ') or \
                       line_stripped.startswith('@'):
                        success_end_idx = j
                        break
            
            if success_end_idx == -1:
                success_end_idx = len(lines)
        
        # Find compute_reward (exterior one, should come after compute_success)
        if 'def compute_reward(' in line and class_ended:
            # Make sure it's after compute_success if we found it
            if success_def_idx != -1 and i <= success_def_idx:
                continue
                
            reward_def_idx = i
            # Look backwards for @torch.jit.script
            for j in range(i-1, max(0, i-10), -1):
                if '@torch.jit.script' in lines[j]:
                    reward_start_idx = j
                    break
            if reward_start_idx == -1:
                reward_start_idx = i
            
            # Find the end of compute_reward function
            for j in range(i+1, len(lines)):
                line_stripped = lines[j].lstrip()
                if line_stripped and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                    if line_stripped.startswith('def ') or line_stripped.startswith('class ') or \
                       line_stripped.startswith('from ') or line_stripped.startswith('import ') or \
                       line_stripped.startswith('@'):
                        reward_end_idx = j
                        break
            
            if reward_end_idx == -1:
                reward_end_idx = len(lines)
            break
    
    if success_start_idx == -1:
        print("Warning: compute_success function not found")
        return code_string
    
    if reward_start_idx == -1:
        print("Warning: exterior compute_reward function not found")
        return code_string
    
    # Extract the compute_reward function content
    reward_function_lines = lines[reward_start_idx:reward_end_idx]
    reward_function_code = '\n'.join(reward_function_lines)
    
    # Parse the compute_reward function to get its signature
    try:
        module = ast.parse(reward_function_code)
        function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        if function_defs:
            function_def = function_defs[0]
            # Construct the signature with self prefix
            args = [arg.arg for arg in function_def.args.args]
            signature = 'compute_success(self.' + ', self.'.join(args) + ')'
        else:
            print("Warning: Could not parse compute_reward function signature")
            return code_string
    except Exception as e:
        print(f"Warning: Error parsing compute_reward signature: {e}")
        return code_string
    
    # Replace compute_success with compute_reward's content (but keep the name as compute_success)
    new_success_lines = []
    for line in reward_function_lines:
        if 'def compute_reward(' in line:
            # Replace function name but keep the signature
            new_success_lines.append(line.replace('def compute_reward(', 'def compute_success('))
        else:
            new_success_lines.append(line)
    
    # Build the new code
    new_lines = (
        ['from typing import Tuple, Dict',
         'import math',
         'import torch',
         'from torch import Tensor'] + 
        lines[:success_start_idx] +  # Everything before compute_success
        new_success_lines +  # New compute_success (with compute_reward's body)
        lines[success_end_idx:reward_start_idx] +  # Everything between the two functions
        lines[reward_start_idx:reward_end_idx] +  # Keep compute_reward as is
        lines[reward_end_idx:]  # Everything after compute_reward
    )
    
    code_with_replaced_success = '\n'.join(new_lines)
    
    # Pattern to find and replace the compute_success method call - MORE FLEXIBLE
    # Matches any assignment pattern: x = compute_success(...) or x, y, z = compute_success(...)
    method_call_pattern = r'^[^\n]*?=\s*compute_success\s*\(.*?\)'

    def replace_method_call(match):
        # Extract indentation
        indent_match = re.match(r'^(\s*)', match.group(0))
        indent = indent_match.group(1) if indent_match else '        '
        
        return f"""{indent}self.gt_rew_buf, _ = {signature}
        self.consecutive_successes[:] = self.gt_rew_buf.mean()"""

    transformed_code = re.sub(method_call_pattern, replace_method_call, code_with_replaced_success, flags=re.MULTILINE | re.DOTALL)    
    # Clean up any extra blank lines that might have been created
    transformed_code = re.sub(r'\n\s*\n\s*\n+', '\n\n', transformed_code)
    
    return transformed_code


def main():
    """
    Main function to demonstrate the transformation.
    Can be used as a standalone script or imported as a module.
    """
    import sys
    
    if len(sys.argv) > 1:
        # Read from file if provided
        input_file = sys.argv[1]
        with open(input_file, 'r') as f:
            code = f.read()
        
        transformed = transform_code(code)
        
        # Write to output file if specified, otherwise print
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
            with open(output_file, 'w') as f:
                f.write(transformed)
            print(f"Transformed code written to {output_file}")
        else:
            print(transformed)
    else:
        print("Usage: python transform_code.py <input_file> [output_file]")
        print("\nOr import and use transform_code(code_string) function")


if __name__ == "__main__":
    main()