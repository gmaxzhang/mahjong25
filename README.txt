This is a project analyzing Mahjong from a statistical perspective.  
This will include simulations and a reinforcement-learning algorithm to play Mahjong.
It may also include a non-ML algorithm that seeks to maximize expected value of a Mahjong play.

In Mahjong, scoring rules are not standardized--each house plays by their own rules.
For the sake of this project, I will mainly focus on the scoring rules used by my family/family friends back home, though
I will also make this adaptable between different scoring rules (you just need to import a dataset with scoring rules
and their criteria).

To understand how the algorithm functions, here are the scoring criteria for my family's Mahjong rules.

For base rules on the types of pieces and sequences, please look at these links (or the remaining rules will not make sense): 

Types of Tiles: https://www.themahjongproject.com/how-to-play/components (only look at "The Tiles" section)
Note that we do use the flower tiles where you will play a flower tile and then draw a tile from t

Tile Sequences/Order of Play: https://www.themahjongproject.com/how-to-play/basics 
(ignore the section about the last 14 tiles in the "Winning the Hand" section)

Winning hands almost always taking the form of 4 sets (pungs/kongs) or runs and one pair of eyes. 
You can either win from drawing a winning piece yourself or from taking a piece that an opponent plays 
to complete your hand. If you win from taking a piece that an opponent plays, you need two "flower points" unless you
have not taken someone else's piece to complete any set or run.

"Flower points" come in various ways. Each flower tile is worth one "flower point" and you (almost) always
play your flowers as soon as you draw them as they don't pair in runs or sets and also allow you to draw a tile
from the back of the wall (opposite side of where you draw from). You can also earn different 
amounts of "flower points" from three-of-a-kinds or four-of-a-kinds (but not eyes/pair) as follows:

For dragons, you get two flower points for a three-of-a-kind, three points for an open set of four, 
and four points for a closed set of four.

For dragons, you get one flower point for a three-of-a-kind, two points for an open set of four, 
and three points for a closed set of four.

For bamboos, characters, and circles, you get zero flower points for a three-of-a-kind, one point for an open set of four, 
and two points for a closed set of four.

For hand scoring, note that a winning hand's default base score is 10. Then, each flower point adds 1 to the hand score. From 
there, there are additional "special hands" that can improve the hand's base score. These hands do not require two flower points.

"One Color" hand: if your hand contains all bamboos, all characters, or all circles, the hand's base score is 40 instead of 10.

Mixed "one-color" hand: if your hand contains four sets that are all bamboos, all characters, or all circles with a pair of eyes
that are wind or dragon pieces, the hand's base score is 20.

All Apart: for each of bamboos, characters, and circles, you have a subset of exactly one of {1, 4, 7}, {2, 5, 8}, or {3, 6, 9}
with the remaining tiles being unique wind, dragon, and/or flowers (at most 1 flower). This is the only time a flower
can be part of a winning hand. This hand has a base score of 20. This is also one of the few hands that doesn't follow the
four sets/runs + one pair of eyes strcture.

Seven Pairs: instead of four sets/runs + one pair of eyes, you get a winning hand from making seven pairs of tiles;
if you have four of a kind, you can separate it into two separate pairs, but you will not get the flower points of the
four of a kind. This hand has a base score of 40.

"Peng-peng-hu": if all of your sets are three-of-a-kind or four-of-a-kind (no runs), the hand's base score is 20.

"Eating hand": if you make all four sets by taking others' discarded tiles, then your winning hand's base score is 20.

Additionally, if you play a flower tile or obtain a four-of-a-kind and pick up your hand-completing tile from the back of the
wall, you get an additional 5 points to your hand.

If your winning hand fulfills multiple of these special conditions, add the base scores together (i.e. "peng-peng-hu" of the
same "color" would be 40 + 20 = 60 points).

In terms of how points are aggregated/distributed, note that this is a zero-sum game. If the winning player obtained their last 
tile from a different player, then they gain the number of points of their hand, but then the player whose tile was taken also 
loses the number of tiles the hand is worth. If the winning player draws their final tile themselves, then everybody else
loses the number of points the winning hand is worth, while the winner gains 3 times as many points as the hand's value. You can
think of it as the loser(s) paying the winner the amount of points the hand is worth, with all opponents being deemed the loser
if the winner drew their own final tile.

As you can see, there are a lot of potential mathematical complexities under this rule set. Is it ever optimal to try
to draw a winning piece yourself to get the triple payout even if it hurts your win probability? If you hold five pairs
but you draw one of the tiles of which you have a pair, do you go for the seven pairs hand or "peng-peng-hu"?