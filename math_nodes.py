import math
import random


class MathOperation:
    """Basic mathematical operations on two numbers"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "b": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "operation": (["+", "-", "*", "/", "//", "%", "**", "min", "max", "avg"],),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("result", "result_int", "expression")
    FUNCTION = "calculate"
    CATEGORY = "math"

    def calculate(self, a, b, operation):
        """Perform mathematical operation"""
        
        try:
            if operation == "+":
                result = a + b
            elif operation == "-":
                result = a - b
            elif operation == "*":
                result = a * b
            elif operation == "/":
                if b == 0:
                    result = 0
                else:
                    result = a / b
            elif operation == "//":
                if b == 0:
                    result = 0
                else:
                    result = a // b
            elif operation == "%":
                if b == 0:
                    result = 0
                else:
                    result = a % b
            elif operation == "**":
                result = a ** b
            elif operation == "min":
                result = min(a, b)
            elif operation == "max":
                result = max(a, b)
            elif operation == "avg":
                result = (a + b) / 2
        except:
            result = 0
            
        result_int = int(round(result))
        expression = f"{a} {operation} {b} = {result}"
        
        return (result, result_int, expression)


class MathFunction:
    """Mathematical functions on a single number"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "function": ([
                    "abs", "sign", "sqrt", "square", "cube",
                    "sin", "cos", "tan", "asin", "acos", "atan",
                    "sinh", "cosh", "tanh",
                    "exp", "log", "log10", "log2",
                    "ceil", "floor", "round", "trunc",
                    "radians", "degrees"
                ],),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("result", "result_int", "expression")
    FUNCTION = "calculate"
    CATEGORY = "math"

    def calculate(self, value, function):
        """Apply mathematical function"""
        
        try:
            if function == "abs":
                result = abs(value)
            elif function == "sign":
                result = 1 if value > 0 else (-1 if value < 0 else 0)
            elif function == "sqrt":
                result = math.sqrt(abs(value))
            elif function == "square":
                result = value * value
            elif function == "cube":
                result = value * value * value
            elif function == "sin":
                result = math.sin(value)
            elif function == "cos":
                result = math.cos(value)
            elif function == "tan":
                result = math.tan(value)
            elif function == "asin":
                result = math.asin(max(-1, min(1, value)))
            elif function == "acos":
                result = math.acos(max(-1, min(1, value)))
            elif function == "atan":
                result = math.atan(value)
            elif function == "sinh":
                result = math.sinh(value)
            elif function == "cosh":
                result = math.cosh(value)
            elif function == "tanh":
                result = math.tanh(value)
            elif function == "exp":
                result = math.exp(value)
            elif function == "log":
                result = math.log(abs(value)) if value != 0 else 0
            elif function == "log10":
                result = math.log10(abs(value)) if value != 0 else 0
            elif function == "log2":
                result = math.log2(abs(value)) if value != 0 else 0
            elif function == "ceil":
                result = math.ceil(value)
            elif function == "floor":
                result = math.floor(value)
            elif function == "round":
                result = round(value)
            elif function == "trunc":
                result = math.trunc(value)
            elif function == "radians":
                result = math.radians(value)
            elif function == "degrees":
                result = math.degrees(value)
        except:
            result = 0
            
        result_int = int(round(result))
        expression = f"{function}({value}) = {result}"
        
        return (result, result_int, expression)


class BooleanLogic:
    """Boolean logic operations"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("BOOLEAN", {"default": False}),
                "b": ("BOOLEAN", {"default": False}),
                "operation": (["AND", "OR", "XOR", "NAND", "NOR", "XNOR"],),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "STRING")
    RETURN_NAMES = ("result", "result_int", "expression")
    FUNCTION = "calculate"
    CATEGORY = "math/logic"

    def calculate(self, a, b, operation):
        """Perform boolean operation"""
        
        if operation == "AND":
            result = a and b
        elif operation == "OR":
            result = a or b
        elif operation == "XOR":
            result = a != b
        elif operation == "NAND":
            result = not (a and b)
        elif operation == "NOR":
            result = not (a or b)
        elif operation == "XNOR":
            result = a == b
            
        result_int = 1 if result else 0
        expression = f"{a} {operation} {b} = {result}"
        
        return (result, result_int, expression)


class CompareNumbers:
    """Compare two numbers"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "b": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "comparison": (["==", "!=", "<", ">", "<=", ">="],),
                "tolerance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Tolerance for equality comparison"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "STRING")
    RETURN_NAMES = ("result", "result_int", "expression")
    FUNCTION = "compare"
    CATEGORY = "math/logic"

    def compare(self, a, b, comparison, tolerance):
        """Compare numbers"""
        
        if comparison == "==":
            result = abs(a - b) <= tolerance
        elif comparison == "!=":
            result = abs(a - b) > tolerance
        elif comparison == "<":
            result = a < b
        elif comparison == ">":
            result = a > b
        elif comparison == "<=":
            result = a <= b
        elif comparison == ">=":
            result = a >= b
            
        result_int = 1 if result else 0
        expression = f"{a} {comparison} {b} = {result}"
        
        return (result, result_int, expression)


class ConvertNumber:
    """Convert between number types and formats"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "conversion": ([
                    "to_int", "to_float", "to_bool",
                    "to_hex", "to_binary", "to_octal",
                    "normalize_0_1", "normalize_-1_1",
                    "denormalize_0_255", "denormalize_0_65535"
                ],),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("float_value", "int_value", "bool_value", "string_value")
    FUNCTION = "convert"
    CATEGORY = "math/convert"

    def convert(self, value, conversion):
        """Convert number to different format"""
        
        float_value = value
        int_value = int(round(value))
        bool_value = value != 0
        string_value = str(value)
        
        if conversion == "to_int":
            int_value = int(round(value))
            float_value = float(int_value)
            string_value = str(int_value)
        elif conversion == "to_float":
            float_value = float(value)
            string_value = str(float_value)
        elif conversion == "to_bool":
            bool_value = value != 0
            int_value = 1 if bool_value else 0
            float_value = float(int_value)
            string_value = str(bool_value)
        elif conversion == "to_hex":
            string_value = hex(int(round(value)))
        elif conversion == "to_binary":
            string_value = bin(int(round(value)))
        elif conversion == "to_octal":
            string_value = oct(int(round(value)))
        elif conversion == "normalize_0_1":
            # Assumes input is 0-255
            float_value = value / 255.0
            int_value = int(round(float_value))
        elif conversion == "normalize_-1_1":
            # Assumes input is 0-255
            float_value = (value / 127.5) - 1.0
            int_value = int(round(float_value))
        elif conversion == "denormalize_0_255":
            # Assumes input is 0-1
            float_value = value * 255.0
            int_value = int(round(float_value))
        elif conversion == "denormalize_0_65535":
            # Assumes input is 0-1
            float_value = value * 65535.0
            int_value = int(round(float_value))
            
        return (float_value, int_value, bool_value, string_value)


class RandomNumber:
    """Generate random numbers"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 999999}),
                "mode": (["float", "int", "gaussian", "choice"],),
            },
            "optional": {
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Standard deviation for gaussian mode"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("float_value", "int_value", "info")
    FUNCTION = "generate"
    CATEGORY = "math/random"

    def generate(self, min_value, max_value, seed, mode, sigma=1.0):
        """Generate random number"""
        
        # Set seed if specified
        if seed >= 0:
            random.seed(seed)
            
        if mode == "float":
            float_value = random.uniform(min_value, max_value)
            int_value = int(round(float_value))
            info = f"Random float: {float_value:.4f}"
        elif mode == "int":
            int_value = random.randint(int(min_value), int(max_value))
            float_value = float(int_value)
            info = f"Random int: {int_value}"
        elif mode == "gaussian":
            mean = (min_value + max_value) / 2
            float_value = random.gauss(mean, sigma)
            # Clamp to range
            float_value = max(min_value, min(max_value, float_value))
            int_value = int(round(float_value))
            info = f"Gaussian: {float_value:.4f} (μ={mean:.2f}, σ={sigma:.2f})"
        elif mode == "choice":
            # Random choice between min and max
            float_value = random.choice([min_value, max_value])
            int_value = int(round(float_value))
            info = f"Choice: {float_value}"
            
        return (float_value, int_value, info)


class Clamp:
    """Clamp value between min and max"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "min_value": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("clamped", "clamped_int", "was_clamped", "info")
    FUNCTION = "clamp"
    CATEGORY = "math"

    def clamp(self, value, min_value, max_value):
        """Clamp value to range"""
        
        original = value
        clamped = max(min_value, min(max_value, value))
        clamped_int = int(round(clamped))
        was_clamped = clamped != original
        
        if was_clamped:
            info = f"Clamped {original:.2f} to {clamped:.2f}"
        else:
            info = f"Value {original:.2f} within range"
            
        return (clamped, clamped_int, was_clamped, info)


class Remap:
    """Remap value from one range to another"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.5, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "in_min": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "in_max": ("FLOAT", {"default": 1.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "out_max": ("FLOAT", {"default": 100.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "clamp_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("remapped", "remapped_int", "info")
    FUNCTION = "remap"
    CATEGORY = "math"

    def remap(self, value, in_min, in_max, out_min, out_max, clamp_output):
        """Remap value from input range to output range"""
        
        # Avoid division by zero
        if in_max - in_min == 0:
            remapped = out_min
        else:
            # Linear interpolation
            normalized = (value - in_min) / (in_max - in_min)
            remapped = out_min + normalized * (out_max - out_min)
            
        if clamp_output:
            remapped = max(out_min, min(out_max, remapped))
            
        remapped_int = int(round(remapped))
        info = f"{value:.2f} [{in_min:.1f}-{in_max:.1f}] → {remapped:.2f} [{out_min:.1f}-{out_max:.1f}]"
        
        return (remapped, remapped_int, info)


class Smooth:
    """Smooth interpolation between values"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "target": ("FLOAT", {"default": 1.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (["linear", "ease_in", "ease_out", "ease_in_out", "cubic", "bounce"],),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("smoothed", "smoothed_int", "info")
    FUNCTION = "smooth"
    CATEGORY = "math"

    def smooth(self, current, target, factor, method):
        """Smooth interpolation"""
        
        t = factor
        
        if method == "linear":
            smoothed = current + (target - current) * t
        elif method == "ease_in":
            smoothed = current + (target - current) * (t * t)
        elif method == "ease_out":
            smoothed = current + (target - current) * (1 - (1 - t) * (1 - t))
        elif method == "ease_in_out":
            if t < 0.5:
                smoothed = current + (target - current) * (2 * t * t)
            else:
                smoothed = current + (target - current) * (1 - pow(-2 * t + 2, 2) / 2)
        elif method == "cubic":
            smoothed = current + (target - current) * (t * t * t)
        elif method == "bounce":
            # Simple bounce effect
            n1 = 7.5625
            d1 = 2.75
            if t < 1 / d1:
                bounce_t = n1 * t * t
            elif t < 2 / d1:
                t -= 1.5 / d1
                bounce_t = n1 * t * t + 0.75
            elif t < 2.5 / d1:
                t -= 2.25 / d1
                bounce_t = n1 * t * t + 0.9375
            else:
                t -= 2.625 / d1
                bounce_t = n1 * t * t + 0.984375
            smoothed = current + (target - current) * bounce_t
            
        smoothed_int = int(round(smoothed))
        info = f"{method}: {current:.2f} → {target:.2f} ({factor*100:.0f}%) = {smoothed:.2f}"
        
        return (smoothed, smoothed_int, info)


class Constants:
    """Mathematical and physical constants"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "constant": ([
                    "pi", "tau", "e", "phi", "sqrt2", "sqrt3",
                    "deg2rad", "rad2deg",
                    "inch2cm", "cm2inch", "ft2m", "m2ft",
                    "lb2kg", "kg2lb", "f2c", "c2f"
                ],),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("value", "description")
    FUNCTION = "get_constant"
    CATEGORY = "math"

    def get_constant(self, constant):
        """Get mathematical constant"""
        
        constants = {
            "pi": (math.pi, "π = 3.14159..."),
            "tau": (math.tau, "τ = 2π = 6.28318..."),
            "e": (math.e, "Euler's number = 2.71828..."),
            "phi": ((1 + math.sqrt(5)) / 2, "Golden ratio = 1.61803..."),
            "sqrt2": (math.sqrt(2), "√2 = 1.41421..."),
            "sqrt3": (math.sqrt(3), "√3 = 1.73205..."),
            "deg2rad": (math.pi / 180, "Degrees to radians"),
            "rad2deg": (180 / math.pi, "Radians to degrees"),
            "inch2cm": (2.54, "Inches to centimeters"),
            "cm2inch": (1 / 2.54, "Centimeters to inches"),
            "ft2m": (0.3048, "Feet to meters"),
            "m2ft": (1 / 0.3048, "Meters to feet"),
            "lb2kg": (0.453592, "Pounds to kilograms"),
            "kg2lb": (1 / 0.453592, "Kilograms to pounds"),
            "f2c": (5/9, "Fahrenheit to Celsius factor"),
            "c2f": (9/5, "Celsius to Fahrenheit factor"),
        }
        
        value, description = constants.get(constant, (0.0, "Unknown"))
        return (value, description)