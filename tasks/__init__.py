"""Ravens tasks."""

from tasks.align_box_corner import AlignBoxCorner
from tasks.align_rope import AlignRope
from tasks.assembling_kits import AssemblingKits
from tasks.assembling_kits_seq import AssemblingKitsSeq
from tasks.block_insertion import BlockInsertion
from tasks.extended_tasks import *
from tasks.generated_task import GeneratedTask
from tasks.manipulating_rope import ManipulatingRope
from tasks.move_blocks import MoveBlocksBetweenAbsolutePositions
from tasks.move_blocks import MoveBlocksBetweenAbsolutePositionsBySize
from tasks.packing_boxes import PackingBoxes
from tasks.packing_boxes_pairs import PackingBoxesPairs
from tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from tasks.packing_shapes import PackingShapes
from tasks.palletizing_boxes import PalletizingBoxes
from tasks.place_red_in_green import PlaceRedInGreen
from tasks.put_block_in_bowl import PutBlockInMismatchingBowl
from tasks.put_block_in_bowl import PutBlockInMatchingBowl
from tasks.separating_piles import SeparatingPiles
from tasks.stack_block_pyramid import StackBlockPyramid
from tasks.stack_block_pyramid_seq import StackBlockPyramidSeq
from tasks.stack_blocks import StackAllBlocksOnAZone
from tasks.stack_blocks import StackBlocksOfSameSize
from tasks.stack_blocks import StackBlockOfSameColor
from tasks.stack_blocks import StackBlocksByColorAndSize
from tasks.stack_blocks import StackBlocksWithAlternateColor
from tasks.stack_blocks import StackBlocksByRelativePositionAndColor
from tasks.sweeping_piles import SweepingPiles
from tasks.task import Task
from tasks.towers_of_hanoi import TowersOfHanoi
from tasks.towers_of_hanoi_seq import TowersOfHanoiSeq

from loho_tasks.build_cube import BuildCubeWithSameColorBlock


names = {
    # demo conditioned
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi,
    'gen-task': GeneratedTask,

    # goal conditioned
    'align-rope': AlignRope,
    'assembling-kits-seq': AssemblingKitsSeq,
    'assembling-kits-seq-seen-colors': AssemblingKitsSeqSeenColors,
    'assembling-kits-seq-unseen-colors': AssemblingKitsSeqUnseenColors,
    'assembling-kits-seq-full': AssemblingKitsSeqFull,
    'packing-shapes': PackingShapes,
    'packing-boxes-pairs': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-seen-colors': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-unseen-colors': PackingBoxesPairsUnseenColors,
    'packing-boxes-pairs-full': PackingBoxesPairsFull,
    'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
    'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
    'packing-seen-google-objects-group': PackingSeenGoogleObjectsGroup,
    'packing-unseen-google-objects-group': PackingUnseenGoogleObjectsGroup,
    'put-block-in-bowl': PutBlockInBowlSeenColors,
    'put-block-in-bowl-seen-colors': PutBlockInBowlSeenColors,
    'put-block-in-bowl-unseen-colors': PutBlockInBowlUnseenColors,
    'put-block-in-bowl-full': PutBlockInBowlFull,
    'stack-block-pyramid-seq': StackBlockPyramidSeqSeenColors,
    'stack-block-pyramid-seq-seen-colors': StackBlockPyramidSeqSeenColors,
    'stack-block-pyramid-seq-unseen-colors': StackBlockPyramidSeqUnseenColors,
    'stack-block-pyramid-seq-full': StackBlockPyramidSeqFull,
    'separating-piles': SeparatingPilesSeenColors,
    'separating-piles-seen-colors': SeparatingPilesSeenColors,
    'separating-piles-unseen-colors': SeparatingPilesUnseenColors,
    'separating-piles-full': SeparatingPilesFull,
    'towers-of-hanoi-seq': TowersOfHanoiSeqSeenColors,
    'towers-of-hanoi-seq-seen-colors': TowersOfHanoiSeqSeenColors,
    'towers-of-hanoi-seq-unseen-colors': TowersOfHanoiSeqUnseenColors,
    'towers-of-hanoi-seq-full': TowersOfHanoiSeqFull,

    'stack-blocks-by-color-and-size': StackBlocksByColorAndSize,
    'stack-all-blocks-on-a-zone': StackAllBlocksOnAZone,
    'stack-blocks-of-same-size': StackBlocksOfSameSize,
    'stack-blocks-of-same-color': StackBlockOfSameColor,
    'stack-blocks-with-alternate-color': StackBlocksWithAlternateColor,
    'stack-blocks-by-relative-position-and-color': StackBlocksByRelativePositionAndColor,
    'move-blocks-between-absolute-positions': MoveBlocksBetweenAbsolutePositions,
    'move-blocks-between-absolute-positions-by-size': MoveBlocksBetweenAbsolutePositionsBySize,
    'put-block-into-mismatching-bowl': PutBlockInMismatchingBowl,
    'put-block-into-matching-bowl': PutBlockInMatchingBowl,

    'build-cube' : BuildCubeWithSameColorBlock
}


# from generated_tasks import new_names
# names.update(new_names)
# from loho_tasks import new_names
# print(new_names)
# names.update(new_names)
