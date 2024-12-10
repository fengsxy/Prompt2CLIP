# data/scene_analyzer.py

from typing import List, Dict, Set
import re
from collections import Counter

class SceneAnalyzer:
    def __init__(self):
        # 场景类型关键词
        self.indoor_keywords = {
            'room', 'kitchen', 'bedroom', 'bathroom', 'living room', 'office',
            'indoor', 'inside', 'restaurant', 'store', 'shop', 'mall'
        }
        
        self.outdoor_keywords = {
            'street', 'park', 'beach', 'mountain', 'forest', 'garden',
            'outdoor', 'outside', 'playground', 'yard', 'road', 'field'
        }
        
        # 时间相关关键词
        self.time_keywords = {
            'day': {'daytime', 'day', 'afternoon', 'morning', 'noon', 'sunny'},
            'night': {'night', 'evening', 'dark', 'nighttime', 'sunset', 'dusk'}
        }
        
        # 天气相关关键词
        self.weather_keywords = {
            'sunny': {'sunny', 'clear', 'bright'},
            'rainy': {'rain', 'rainy', 'raining', 'wet'},
            'cloudy': {'cloudy', 'overcast', 'gray'},
            'snowy': {'snow', 'snowy', 'snowing'},
        }
        
        # 环境类型关键词
        self.environment_keywords = {
            'urban': {'city', 'urban', 'downtown', 'street', 'building'},
            'rural': {'rural', 'countryside', 'farm', 'field', 'nature'},
            'suburban': {'suburban', 'suburb', 'residential', 'neighborhood'}
        }

    def _check_keywords(self, text: str, keywords: Set[str]) -> bool:
        """检查文本中是否包含关键词"""
        text = text.lower()
        return any(keyword in text for keyword in keywords)

    def _get_majority_vote(self, captions: List[str], keyword_dict: Dict[str, Set[str]]) -> str:
        """使用多数投票确定类别"""
        votes = Counter()
        for caption in captions:
            caption = caption.lower()
            for category, keywords in keyword_dict.items():
                if self._check_keywords(caption, keywords):
                    votes[category] += 1
        
        if not votes:
            return 'unknown'
        
        return votes.most_common(1)[0][0]

    def extract_scene_info(self, captions: List[str]) -> Dict[str, str]:
        """从图像描述中提取场景信息"""
        scene_info = {}
        
        # 确定室内/室外
        indoor_count = sum(self._check_keywords(cap, self.indoor_keywords) for cap in captions)
        outdoor_count = sum(self._check_keywords(cap, self.outdoor_keywords) for cap in captions)
        
        scene_info['location_type'] = 'indoor' if indoor_count > outdoor_count else 'outdoor'
        
        # 确定时间
        scene_info['time'] = self._get_majority_vote(captions, self.time_keywords)
        
        # 确定天气
        scene_info['weather'] = self._get_majority_vote(captions, self.weather_keywords)
        
        # 确定环境类型
        scene_info['environment'] = self._get_majority_vote(captions, self.environment_keywords)
        
        return scene_info

    def generate_scene_description(self, captions: List[str]) -> str:
        """生成场景描述"""
        scene_info = self.extract_scene_info(captions)
        
        description_parts = []
        
        # 添加环境类型
        if scene_info['environment'] != 'unknown':
            description_parts.append(f"in a {scene_info['environment']} area")
            
        # 添加室内/室外信息
        description_parts.append(scene_info['location_type'])
        
        # 添加时间信息
        if scene_info['time'] != 'unknown':
            description_parts.append(f"during {scene_info['time']}")
            
        # 添加天气信息
        if scene_info['weather'] != 'unknown':
            description_parts.append(f"on a {scene_info['weather']} day")
            
        # 组合描述
        description = ' '.join(description_parts)
        
        return description

    def analyze_spatial_relationships(self, captions: List[str]) -> List[str]:
        """分析空间关系"""
        spatial_patterns = [
            r'(in front of|behind|next to|beside|near|on top of|above|below|under|inside|outside) (the|a|an) (\w+)',
            r'(\w+) (is|are) (in front of|behind|next to|beside|near|on top of|above|below|under|inside|outside) (the|a|an) (\w+)'
        ]
        
        relationships = []
        for caption in captions:
            for pattern in spatial_patterns:
                matches = re.finditer(pattern, caption, re.IGNORECASE)
                for match in matches:
                    relationships.append(match.group(0))
                    
        return list(set(relationships))  # 去重

    def get_action_context(self, captions: List[str]) -> List[str]:
        """提取动作上下文"""
        action_patterns = [
            r'(\w+ing) (the|a|an) (\w+)',
            r'(\w+s) (the|a|an) (\w+)',
            r'is (\w+ing)',
            r'are (\w+ing)'
        ]
        
        actions = []
        for caption in captions:
            for pattern in action_patterns:
                matches = re.finditer(pattern, caption, re.IGNORECASE)
                for match in matches:
                    actions.append(match.group(0))
                    
        return list(set(actions))  # 去重

# 使用示例
def example_usage():
    analyzer = SceneAnalyzer()
    
    # 测试数据
    captions = [
        "A man is skateboarding down a path and a dog is running by his side.",
        "A man on a skateboard with a dog outside on a sunny day.",
        "A person riding a skate board with a dog following beside in an urban area.",
        "This man is riding a skateboard behind a dog in the daytime.",
        "A man walking his dog on a quiet suburban road."
    ]
    
    # 提取场景信息
    scene_info = analyzer.extract_scene_info(captions)
    print("Scene Information:")
    print(scene_info)
    
    # 生成场景描述
    description = analyzer.generate_scene_description(captions)
    print("\nGenerated Scene Description:")
    print(description)
    
    # 分析空间关系
    spatial = analyzer.analyze_spatial_relationships(captions)
    print("\nSpatial Relationships:")
    print(spatial)
    
    # 获取动作上下文
    actions = analyzer.get_action_context(captions)
    print("\nAction Context:")
    print(actions)

if __name__ == "__main__":
    example_usage()